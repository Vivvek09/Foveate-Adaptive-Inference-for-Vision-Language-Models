#!/usr/bin/env bash
# Run every stage end to end.
#   MODE=smoke  bash scripts/run_all.sh   # first GPU check: 8 questions, 1-2 configs per model
#   MODE=pilot  GPUS=2 bash scripts/run_all.sh   # fixed-seed 30% of the questions
#   MODE=full   GPUS=2 bash scripts/run_all.sh   # everything; re-run to resume
set -euo pipefail
MODE=${MODE:-pilot}
GPUS=${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
GPUS=${GPUS:-1}
TASKS=${TASKS:-all}           # all | lite (skips the 11 GB Perception Test zip)
RUN_7B=${RUN_7B:-1}
RUN_SHIFT=${RUN_SHIFT:-1}
RUN_LINGOQA=${RUN_LINGOQA:-1}
RUN_SERVING=${RUN_SERVING:-1}
echo "mode=$MODE gpus=$GPUS tasks=$TASKS 7b=$RUN_7B"
MODELS="3b"; [[ $RUN_7B == 1 ]] && MODELS="3b 7b"

mkdir -p "${FOVEATE_RESULTS:-results}"
foveate env > "${FOVEATE_RESULTS:-results}/env.json" || true

# Download model weights once, before parallel workers race for them
python -c "
from huggingface_hub import snapshot_download as d
for m in ['Qwen/Qwen2.5-VL-3B-Instruct'] + (['Qwen/Qwen2.5-VL-7B-Instruct-AWQ'] if '$RUN_7B' == '1' else []):
    d(m)"

# Always first: the smoke test (generated videos, no download). Stops here on failure.
foveate smoke --models $MODELS --profile
if [[ "$MODE" == "smoke" ]]; then
  foveate report
  echo "smoke done: see ${FOVEATE_RESULTS:-results}/smoke/SMOKE.md"
  exit 0
fi

# Stage 1: profile (one GPU is enough; synthetic frames, no dataset needed)
foveate profile --model 3b --source synthetic

# Stage 2: quality grid on MVBench
foveate download mvbench --tasks "$TASKS"
for m in $MODELS; do foveate grid --dataset mvbench --$MODE --tasks "$TASKS" --model $m --gpus "$GPUS"; done

# Stage 4 data: shifted copies (fog, night, blur) for the shift configs
if [[ "$RUN_SHIFT" == "1" ]]; then
  for m in $MODELS; do foveate shift --dataset mvbench --$MODE --tasks "$TASKS" --model $m --gpus "$GPUS"; done
fi

# Stages 3-4: replay policies and shift from the cache (CPU, seconds)
foveate replay --dataset mvbench --$MODE --tasks "$TASKS"

# Driving demo: LingoQA (needs Google Drive access; skipped on failure)
if [[ "$RUN_LINGOQA" == "1" ]]; then
  if foveate download lingoqa; then
    for m in $MODELS; do foveate grid --dataset lingoqa --$MODE --model $m --gpus "$GPUS"; done
    foveate judge --dataset lingoqa --$MODE
    foveate replay --dataset lingoqa --$MODE
  else
    echo "LingoQA download failed; see README (manual download)"
  fi
fi

# Stage 5: serving under load
if [[ "$RUN_SERVING" == "1" ]]; then
  MODE=$MODE TASKS=$TASKS SERVE_7B=$RUN_7B bash scripts/serve_and_load.sh
fi

foveate report
echo "done: see ${FOVEATE_RESULTS:-results}/REPORT.md"
