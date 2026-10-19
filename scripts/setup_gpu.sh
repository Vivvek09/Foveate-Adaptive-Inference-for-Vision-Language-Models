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

# Dependency doctor. Kaggle images ship torchaudio/torchcodec/torchvision built for their own
# torch; once vLLM swaps torch, those builds can fail to load. Foveate uses none of
# torchaudio/torchcodec (frames go in as PIL images, no audio), but transformers imports
# torchaudio at module level whenever it is installed, which breaks AutoProcessor with
# "Could not import module 'AutoProcessor'". Test each in a fresh interpreter and repair.
imports_ok() { python -c "import $1" >/tmp/foveate_import_$1.log 2>&1; }
for mod in torchaudio torchcodec; do
  if python -c "import importlib.util,sys; sys.exit(importlib.util.find_spec('$mod') is None)" && ! imports_ok "$mod"; then
    echo "doctor: $mod is installed but does not import with this torch:"
    tail -1 "/tmp/foveate_import_$mod.log"
    echo "doctor: uninstalling $mod (unused by Foveate; transformers works without it)"
    pip uninstall -y -q "$mod"
  fi
done
if ! imports_ok torchvision; then
  echo "doctor: torchvision does not import with this torch:"; tail -1 /tmp/foveate_import_torchvision.log
  TV=$(python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'0.{int(v[1])+15}.*')")
  CU=$(python -c "import torch; print('cu' + (torch.version.cuda or '').replace('.', ''))")
  echo "doctor: installing torchvision==${TV} for ${CU}"
  pip install -q --no-deps "torchvision==${TV}" --index-url "https://download.pytorch.org/whl/${CU}" \
    || pip install -q --no-deps "torchvision==${TV}"
fi
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda ok:", torch.cuda.is_available(),
      "gpus:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

def root(e):
    while (e.__cause__ or e.__context__) is not None:
        e = e.__cause__ or e.__context__
    return f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"

ok = True
for mod, names in [("transformers", ["AutoProcessor", "Qwen2_5_VLForConditionalGeneration"]),
                   ("vllm", ["LLM", "SamplingParams"])]:
    try:
        m = importlib.import_module(mod)
        for n in names:
            getattr(m, n)
        print(mod, m.__version__, "imports ok")
    except Exception as e:
        ok = False
        print(f"ERROR: {mod} does not import: {root(e)}")
if not ok:
    print("The smoke test will fail until this import works; send this output.")
PY
