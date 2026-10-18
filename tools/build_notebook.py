"""Build notebooks/foveate_kaggle.ipynb: one self-contained notebook that runs everything.

The notebook embeds the package source (so it works without cloning the repo),
installs it, and runs every stage with a single MODE switch. Rebuild it after
changing the code:  python tools/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMBED_GLOBS = ["pyproject.toml", "README.md", "LICENSE", "src/foveate/**/*.py",
               "src/foveate/configs/*.yaml", "scripts/*.sh", "theory/*.md"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def embedded_files() -> dict[str, str]:
    files = {}
    for pat in EMBED_GLOBS:
        for p in sorted(ROOT.glob(pat)):
            if "__pycache__" in p.parts:
                continue
            files[str(p.relative_to(ROOT))] = p.read_text()
    return files


def build() -> dict:
    files = embedded_files()
    cells = [
        md("""
# Foveate on Kaggle: profile VLM inference, then spend compute only where it is needed

**Before you run:** *Settings → Accelerator → GPU T4 x2* and *Settings → Internet → On*.
Then set `MODE` below and use *Run All*. Results land in `/kaggle/working/foveate_results`
(kept after the session ends) and are zipped at the end for download.

| Stage | What it does | Theory |
|---|---|---|
| 1 profile | where batch-1 latency goes (decode, ViT, prefill, decode) | `theory/01` |
| 2 grid | every question × every frames/resolution/model config, cached | `theory/02` |
| 3 replay | static vs rules vs cascade vs oracle, held-out | `theory/03` |
| 4 shift | fog/night/blur: does the cascade spend more on harder inputs? | `theory/04` |
| 5 serve | vLLM (3B on GPU0, 7B-AWQ on GPU1) + gateway under Poisson load | `theory/05` |

`MODE = "pilot"` runs a fixed-seed 30% of the questions (stratified by task, same
questions every time). `MODE = "full"` runs everything. Every GPU step is resumable:
if the session dies, re-run the notebook and finished questions are skipped (as long
as `/kaggle/working/foveate_results` was kept, e.g. via *Save Version*).
"""),
        code("""
# ---- settings -------------------------------------------------------------
MODE = "pilot"          # "pilot" (fixed 30%) or "full"
TASKS = "auto"          # "all", "lite" (skips the 11 GB Perception zip), or "auto" (lite if disk < 45 GB free)
RUN_PROFILE = True
RUN_7B = True           # the escalation model (needs vLLM)
RUN_SHIFT = True        # stage 4; the longest step in full mode
RUN_LINGOQA = True      # driving demo (Google Drive download; skipped on failure)
RUN_SERVING = True      # stage 5
GPUS = None             # None = all visible GPUs

import os, shutil, subprocess
WORK = "/kaggle/working/foveate" if os.path.isdir("/kaggle") else os.path.abspath("foveate")
os.environ["FOVEATE_RESULTS"] = "/kaggle/working/foveate_results" if os.path.isdir("/kaggle") else os.path.abspath("foveate_results")
os.environ["FOVEATE_DATA"] = "/tmp/foveate_data"        # big, not kept after the session
os.environ["FOVEATE_CACHE"] = "/tmp/foveate_data/frame_cache"
os.makedirs(os.environ["FOVEATE_DATA"], exist_ok=True)
free_gb = shutil.disk_usage("/tmp").free / 2**30
if TASKS == "auto":
    TASKS = "all" if free_gb > 45 else "lite"
if GPUS is None:
    GPUS = max(1, len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()))
print(f"MODE={MODE} TASKS={TASKS} GPUS={GPUS} free disk in /tmp: {free_gb:.0f} GB")
"""),
        code("""
# Optional: a Hugging Face token from Kaggle secrets (Add-ons -> Secrets, name HF_TOKEN)
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF token loaded")
except Exception:
    print("no HF token (fine for public models and MVBench)")
"""),
        md("## 0 · Write the package and install it\nThe next cell writes the Foveate source (identical to the GitHub repo) into `/kaggle/working/foveate`."),
        code("FILES = " + json.dumps(files, indent=0, ensure_ascii=False) + """

for rel, text in FILES.items():
    path = os.path.join(WORK, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
os.chdir(WORK)
print(f"wrote {len(FILES)} files to {WORK}")
"""),
        code("""
# Installs a vLLM build matching the GPU driver, then foveate itself (~5 min).
!bash scripts/setup_gpu.sh
!foveate env
"""),
        code("""
# Download weights once, before two GPU workers race for them.
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-VL-3B-Instruct"] + (["Qwen/Qwen2.5-VL-7B-Instruct-AWQ"] if RUN_7B else []):
    snapshot_download(m)
print("weights cached")
"""),
        md("## Smoke test\n24 generated questions through the 3B model on one GPU. If this fails, fix it before the long runs (see README > Troubleshooting). The log line `first-batch accuracy` should be well above chance."),
        code("!foveate grid --dataset synthetic --full --model 3b --probe 2"),
        md("""
## 1 · Profile: where does the time go?

Batch-1 Transformers runs with CUDA events around the vision encoder and every decode
step, over frames × resolution on synthetic frames (compute does not depend on pixel
content). Read `theory/01_anatomy_of_vlm_inference.md` alongside.
"""),
        code("""
if RUN_PROFILE:
    !foveate profile --model 3b --source synthetic
"""),
        code("""
import json, pandas as pd
R = os.environ["FOVEATE_RESULTS"]
p = f"{R}/profile/3b_synthetic.summary.json"
if os.path.exists(p):
    df = pd.DataFrame(json.load(open(p)))
    cols = ["sweep","frames","res","out_tokens","visual_tokens","load_ms","preprocess_ms","vision_ms","prefill_ms","decode_ms","tpot_ms","e2e_ms","peak_mem_gb","vision_mfu","prefill_mfu"]
    display(df[[c for c in cols if c in df]].round(2))
"""),
        md("## 2 · Quality grid on MVBench\nEvery question through every config once; answers, confidences, tokens and costs are cached under `results/mvbench_<mode>/`. See `theory/02`."),
        code("!foveate download mvbench --tasks {TASKS}"),
        code("!foveate grid --dataset mvbench --{MODE} --tasks {TASKS} --model 3b --gpus {GPUS}"),
        code("""
if RUN_7B:
    !foveate grid --dataset mvbench --{MODE} --tasks {TASKS} --model 7b --gpus {GPUS}
"""),
        md("## 3 + 4 · Shifted copies, then replay every policy\n`shift` reruns the ladder on fog / night / blur frames. `replay` evaluates static configs, learned rules, the confidence cascade, rules+cascade and the oracle on a held-out half, plus the drift study. See `theory/03` and `theory/04`."),
        code("""
if RUN_SHIFT:
    !foveate shift --dataset mvbench --{MODE} --tasks {TASKS} --model 3b --gpus {GPUS}
    if RUN_7B:
        !foveate shift --dataset mvbench --{MODE} --tasks {TASKS} --model 7b --gpus {GPUS}
"""),
        code("""
!foveate replay --dataset mvbench --{MODE} --tasks {TASKS}
rep = json.load(open(f"{R}/mvbench_{MODE}/replay/summary.json"))
print("reference:", rep["reference"], "| ladder:", " -> ".join(rep["ladder"]), "| tau:", round(rep["tau"], 3))
display(pd.DataFrame(rep["policies"])[["policy","accuracy","acc_lo","acc_hi","mean_cost","cost_saving","acc_diff","acc_diff_lo","acc_diff_hi","escalation_rate"]].round(3))
display(pd.DataFrame(rep["confidence"]).round(3))
"""),
        md("## Driving demo · LingoQA\n500 driving-scene questions with free-form answers scored by Lingo-Judge. Google Drive sometimes rate-limits the download; the notebook skips this part if it fails."),
        code("""
if RUN_LINGOQA:
    ok = subprocess.run(["foveate", "download", "lingoqa"]).returncode == 0
    if ok:
        !foveate grid --dataset lingoqa --{MODE} --model 3b --gpus {GPUS}
        if RUN_7B:
            !foveate grid --dataset lingoqa --{MODE} --model 7b --gpus {GPUS}
        !foveate judge --dataset lingoqa --{MODE}
        !foveate replay --dataset lingoqa --{MODE}
    else:
        print("LingoQA download failed: see README > Troubleshooting")
"""),
        md("## 5 · Serving under load\nOne vLLM server per T4 (3B on GPU 0, 7B-AWQ on GPU 1), the Foveate gateway in front, and an open-loop Poisson load test comparing the cascade with the best fixed config. See `theory/05`."),
        code("""
if RUN_SERVING:
    os.environ["MODE"], os.environ["TASKS"] = MODE, TASKS
    !bash scripts/serve_and_load.sh
"""),
        md("## Report"),
        code("""
!foveate report
from IPython.display import Image, Markdown
display(Markdown(open(f"{R}/REPORT.md").read().split("![")[0][:20000]))
import glob
for f in sorted(glob.glob(f"{R}/figures/*.png")):
    print(os.path.basename(f)); display(Image(filename=f, width=720))
"""),
        code("""
shutil.make_archive("/kaggle/working/foveate_results", "zip", R) if os.path.isdir("/kaggle") else None
print("download foveate_results.zip from the Output tab; copy results/ into your GitHub repo")
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = ROOT / "notebooks" / "foveate_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
