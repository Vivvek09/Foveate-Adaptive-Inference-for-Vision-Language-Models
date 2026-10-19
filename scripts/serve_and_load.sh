#!/usr/bin/env bash
# Stage 5: one vLLM server per GPU, the Foveate gateway in front, then Poisson load tests.
#   3B on GPU 0 (:8001), 7B-AWQ on GPU 1 (:8002). Each server owns one GPU's memory;
#   nothing is shared between the two T4s. With one GPU (or SERVE_7B=0) only the 3B
#   server starts and the gateway drops the 7B rungs (logged, not substituted).
set -euo pipefail
MODE=${MODE:-pilot}
TASKS=${TASKS:-all}
RATES=${RATES:-"0.5 1 2"}
N=${N:-200}
MAXRES=${MAXRES:-672}
SERVE_7B=${SERVE_7B:-1}
RES=${FOVEATE_RESULTS:-results}
mkdir -p "$RES/serving"
MMKW="{\"min_pixels\": 784, \"max_pixels\": $((MAXRES * MAXRES))}"
NGPU=$(nvidia-smi -L | wc -l)
[[ $NGPU -lt 2 ]] && { echo "only $NGPU GPU: serving the 3B model only (7B rungs dropped)"; SERVE_7B=0; }
export VLLM_SERVER_DEV_MODE=1   # enables /reset_prefix_cache between policies (best effort)

CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8001 \
  --dtype half --max-model-len 8192 --gpu-memory-utilization 0.88 \
  --limit-mm-per-prompt '{"video": 1, "image": 0}' --mm-processor-kwargs "$MMKW" \
  --enable-prefix-caching > "$RES/serving/vllm_3b.log" 2>&1 &
P1=$!
P2=""
PORTS="8001"
BACKENDS="3b=http://localhost:8001"
if [[ $SERVE_7B == 1 ]]; then
  CUDA_VISIBLE_DEVICES=1 nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ --port 8002 \
    --dtype half --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.88 \
    --limit-mm-per-prompt '{"video": 1, "image": 0}' --mm-processor-kwargs "$MMKW" \
    --enable-prefix-caching > "$RES/serving/vllm_7b.log" 2>&1 &
  P2=$!
  PORTS="8001 8002"
  BACKENDS="$BACKENDS 7b=http://localhost:8002"
fi
trap 'kill $P1 $P2 ${P3:-} 2>/dev/null || true' EXIT

for port in $PORTS; do
  echo "waiting for vLLM on :$port"
  for i in $(seq 1 180); do curl -sf localhost:$port/health >/dev/null && break; sleep 5; done
  curl -sf localhost:$port/health >/dev/null || { echo "vLLM :$port failed; see $RES/serving/vllm_*.log"; exit 1; }
done
# Evidence of GPU placement: which process holds memory on which GPU.
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_bus_id --format=csv \
  > "$RES/serving/gpu_placement.csv" || true
nvidia-smi --query-gpu=index,pci.bus_id,memory.used,memory.total --format=csv \
  >> "$RES/serving/gpu_placement.csv" || true

RUN=$RES/mvbench_${MODE}
POLICY_ARG=()
[[ -f $RUN/replay/policy.json ]] && POLICY_ARG=(--policy-file "$RUN/replay/policy.json")
nohup foveate serve --dataset mvbench --$MODE --tasks "$TASKS" "${POLICY_ARG[@]}" \
  --backends $BACKENDS > "$RES/serving/gateway.log" 2>&1 &
P3=$!
for i in $(seq 1 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 2; done
curl -sf localhost:8000/health | tee "$RES/serving/gateway_health.json"; echo

# Static reference: the replay's reference config, if its model is being served.
REF=${REF:-$(python - <<PY
import json, os
p = "$RUN/replay/summary.json"
print(json.load(open(p))["reference"] if os.path.exists(p) else "3b:8x448")
PY
)}
POLICIES=("cascade")
if [[ $REF == 7b:* && $SERVE_7B != 1 ]]; then
  echo "SKIPPED static:$REF (no 7B server); comparing cascade only" | tee -a "$RES/serving/skipped.txt"
else
  POLICIES+=("static:$REF")
fi

k=0
for rate in $RATES; do
  # Alternate the order per rate so no policy always runs on a warmer server.
  if (( k % 2 )); then ORDER=("${POLICIES[@]}"); else ORDER=($(printf '%s\n' "${POLICIES[@]}" | tac)); fi
  for pol in "${ORDER[@]}"; do
    for port in $PORTS; do curl -sf -X POST localhost:$port/reset_prefix_cache >/dev/null || true; done
    foveate loadtest --dataset mvbench --$MODE --tasks "$TASKS" --policy "$pol" --rate $rate --n $N
  done
  k=$((k + 1))
done
