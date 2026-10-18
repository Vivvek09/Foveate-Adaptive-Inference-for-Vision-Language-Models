#!/usr/bin/env bash
# Run every stage end to end.  MODE=pilot (default, fixed 30%) or MODE=full.
#   MODE=pilot GPUS=2 TASKS=all bash scripts/run_all.sh
set -euo pipefail
MODE=${MODE:-pilot}
GPUS=${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
GPUS=${GPUS:-1}
TASKS=${TASKS:-all}           # all | lite (skips the 11 GB Perception Test zip)
RUN_LINGOQA=${RUN_LINGOQA:-1}
RUN_SERVING=${RUN_SERVING:-1}
echo "mode=$MODE gpus=$GPUS tasks=$TASKS"

foveate env > results_env.json || true

# Download model weights once, before parallel workers race for them
python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ['Qwen/Qwen2.5-VL-3B-Instruct','Qwen/Qwen2.5-VL-7B-Instruct-AWQ']]"

# One-minute smoke test of the GPU stack on 24 generated questions
foveate grid --dataset synthetic --full --model 3b --probe 2

# Stage 1: profile (one GPU is enough; synthetic frames, no dataset needed)
foveate profile --model 3b --source synthetic

# Stage 2: quality grid on MVBench, both models
foveate download mvbench --tasks "$TASKS"
foveate grid --dataset mvbench --$MODE --tasks "$TASKS" --model 3b --gpus "$GPUS"
foveate grid --dataset mvbench --$MODE --tasks "$TASKS" --model 7b --gpus "$GPUS"

# Stage 4 data: shifted copies (fog, night, blur) for the ladder configs
foveate shift --dataset mvbench --$MODE --tasks "$TASKS" --model 3b --gpus "$GPUS"
foveate shift --dataset mvbench --$MODE --tasks "$TASKS" --model 7b --gpus "$GPUS"

# Stages 3-4: replay policies and shift from the cache (CPU, seconds)
foveate replay --dataset mvbench --$MODE --tasks "$TASKS"

# Driving demo: LingoQA (needs Google Drive access; skipped on failure)
if [[ "$RUN_LINGOQA" == "1" ]]; then
  if foveate download lingoqa; then
    foveate grid --dataset lingoqa --$MODE --model 3b --gpus "$GPUS"
    foveate grid --dataset lingoqa --$MODE --model 7b --gpus "$GPUS"
    foveate judge --dataset lingoqa --$MODE
    foveate replay --dataset lingoqa --$MODE
  else
    echo "LingoQA download failed; see README (manual download)"
  fi
fi

# Stage 5: serving under load
if [[ "$RUN_SERVING" == "1" ]]; then
  MODE=$MODE TASKS=$TASKS bash scripts/serve_and_load.sh
fi

foveate report
echo "done: see ${FOVEATE_RESULTS:-results}/REPORT.md"
