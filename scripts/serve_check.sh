#!/usr/bin/env bash
# Quick serving diagnosis (about 3 minutes): start only the 3B vLLM server with the same flags
# as serve_and_load.sh, send one gateway-style request plus variants of it, print vLLM's answers.
set -euo pipefail
RES=${FOVEATE_RESULTS:-results}
mkdir -p "$RES/serving"
MAXRES=${MAXRES:-672}
MMKW="{\"min_pixels\": 784, \"max_pixels\": $((MAXRES * MAXRES))}"
CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 8001 \
  --dtype half --max-model-len 8192 --gpu-memory-utilization 0.88 \
  --limit-mm-per-prompt '{"video": 1, "image": 0}' --mm-processor-kwargs "$MMKW" \
  --enable-prefix-caching > "$RES/serving/vllm_3b_check.log" 2>&1 &
P=$!
trap 'kill $P 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -sf localhost:8001/health >/dev/null && break; sleep 5; done
curl -sf localhost:8001/health >/dev/null || { echo "vLLM did not start"; tail -n 30 "$RES/serving/vllm_3b_check.log"; exit 1; }
python -m foveate.serve.probe --url http://localhost:8001
echo "--- vLLM log, error lines:"
grep -iE "error|exception|value" "$RES/serving/vllm_3b_check.log" | grep -v "Attempted to load weight" | tail -n 20 || true
