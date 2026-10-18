#!/usr/bin/env bash
# Stage 5: start one vLLM server per GPU, the Foveate gateway, then run the load test.
set -euo pipefail
MODE=${MODE:-pilot}
TASKS=${TASKS:-all}
RATES=${RATES:-"0.5 1 2"}
N=${N:-200}
MAXRES=${MAXRES:-672}
RES=${FOVEATE_RESULTS:-results}
mkdir -p "$RES/serving"
MMKW="{\"min_pixels\": 784, \"max_pixels\": $((MAXRES * MAXRES))}"
GPU7B=$(( $(nvidia-smi -L | wc -l) > 1 ? 1 : 0 ))
MEM=$([[ $GPU7B == 0 ]] && echo 0.42 || echo 0.88)   # share one GPU if only one exists

CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8001 \
  --dtype half --max-model-len 8192 --gpu-memory-utilization $MEM \
  --limit-mm-per-prompt '{"video": 1, "image": 0}' --mm-processor-kwargs "$MMKW" \
  --enable-prefix-caching > $RES/serving/vllm_3b.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=$GPU7B nohup vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ --port 8002 \
  --dtype half --quantization awq --max-model-len 8192 --gpu-memory-utilization $MEM \
  --limit-mm-per-prompt '{"video": 1, "image": 0}' --mm-processor-kwargs "$MMKW" \
  --enable-prefix-caching > $RES/serving/vllm_7b.log 2>&1 &
P2=$!
trap 'kill $P1 $P2 ${P3:-} 2>/dev/null || true' EXIT

for port in 8001 8002; do
  echo "waiting for vLLM on :$port"
  for i in $(seq 1 180); do curl -sf localhost:$port/health >/dev/null && break; sleep 5; done
  curl -sf localhost:$port/health >/dev/null || { echo "vLLM :$port failed; see $RES/serving/*.log"; exit 1; }
done

POLICY=$RES/mvbench_${MODE}/replay/policy.json
nohup foveate serve --dataset mvbench --$MODE --tasks "$TASKS" --policy-file "$POLICY" \
  > $RES/serving/gateway.log 2>&1 &
P3=$!
for i in $(seq 1 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 2; done

REF=$(python -c "import json;print(json.load(open('$RES/mvbench_${MODE}/replay/summary.json'))['reference'])")
for rate in $RATES; do
  foveate loadtest --dataset mvbench --$MODE --tasks "$TASKS" --policy cascade --rate $rate --n $N
  foveate loadtest --dataset mvbench --$MODE --tasks "$TASKS" --policy "static:$REF" --rate $rate --n $N
done
