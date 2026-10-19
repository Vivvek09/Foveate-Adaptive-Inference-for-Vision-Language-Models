#!/usr/bin/env bash
# Install Foveate plus a vLLM build that matches this machine's NVIDIA driver.
# Usage: bash scripts/setup_gpu.sh            (from the repo root)
set -euo pipefail

# CUDA version the driver supports, e.g. "12.8" or "13.0".
DRV_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
echo "driver supports CUDA ${DRV_CUDA:-unknown}"
major=${DRV_CUDA%%.*}; minor=${DRV_CUDA##*.}
if [[ -z "${DRV_CUDA}" ]]; then
  echo "no NVIDIA GPU found; installing CPU-only extras"; VLLM_SPEC=""
elif (( major >= 13 )); then
  VLLM_SPEC="vllm==0.32.0"          # API the code was written against (torch cu130 wheels)
elif (( major == 12 && minor >= 8 )); then
  VLLM_SPEC="vllm==0.11.0"          # torch 2.8 / cu128 wheels
else
  VLLM_SPEC="vllm==0.9.2"           # torch 2.7 / cu126 wheels
fi
# Override with: VLLM_OVERRIDE="vllm==X.Y.Z" bash scripts/setup_gpu.sh
VLLM_SPEC=${VLLM_OVERRIDE:-$VLLM_SPEC}

if [[ -n "${VLLM_SPEC}" ]]; then
  echo "installing ${VLLM_SPEC}"
  pip install -q "${VLLM_SPEC}" || echo "WARNING: vLLM install failed; grids with --backend auto fall back to Transformers (labelled backend=hf), serving is unavailable"
fi
pip install -q -e ".[data,hf,serve,dev]"
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda ok:", torch.cuda.is_available(),
      "gpus:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
try:
    v = importlib.import_module("vllm"); print("vllm", v.__version__)
except Exception as e:
    print("vllm not importable:", e)
PY
