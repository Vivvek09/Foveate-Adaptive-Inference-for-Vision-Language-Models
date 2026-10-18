# Foveate

**Spend visual compute only where the question needs it.**

Your eye's fovea sees sharply only where you look, and your brain moves it only when
something needs a closer look. Foveate does the same for vision-language models: it
answers every video question cheaply first (few frames, low resolution, small model)
and escalates to more frames, more pixels or a bigger model only when the cheap
answer looks unsure. It also measures where VLM inference time actually goes, so
every decision rests on profiler numbers rather than guesses.

```
            ┌──────────── Stage 1: profile ────────────┐
 question   │ load │ preprocess │ ViT │ prefill │ decode │  where does the time go?
 + video ──►└───────────────────────────────────────────┘
            Stage 2: grid    every question × every (frames, resolution, model) config, cached
            Stage 3: policy  static vs rules vs confidence cascade vs oracle, held-out evaluation
            Stage 4: shift   fog / night / blur: does the cascade spend more where inputs get harder?
            Stage 5: serve   gateway + vLLM (one model per GPU) under Poisson load: p50/p95/p99
```

| | |
|---|---|
| Models | Qwen2.5-VL-3B-Instruct (cheap), Qwen2.5-VL-7B-Instruct-AWQ (escalation) |
| Main benchmark | **MVBench**: 19 temporal tasks, 3,800 multiple-choice questions (published Qwen2.5-VL scores: 3B 67.0, 7B 69.6) |
| Driving demo | **LingoQA** evaluation set: 500 driving-scene questions, scored with Lingo-Judge |
| Target hardware | Kaggle **2× T4** (16 GB each, fp16). Any CUDA GPU works. |
| Engines | vLLM for throughput and serving; Hugging Face Transformers for per-stage timing |

**Why MVBench:** it is in the video table of almost every recent VLM paper, so your
harness can be checked against a published number, and each answer is a single
option letter, so a full pass over one config takes minutes. See
[theory/02](theory/02_benchmarks_confidence_and_statistics.md).

---

## Two ways to run it

1. **Kaggle notebook (easiest):** open [`notebooks/foveate_kaggle.ipynb`](notebooks/foveate_kaggle.ipynb)
   on Kaggle, set the accelerator to **GPU T4 x2** and **Internet on**, choose
   `MODE = "pilot"` or `"full"` in the first cell, and *Run All*. The notebook embeds
   the whole package, so it needs nothing else.
2. **This repo (any GPU machine):** the commands below.

### Pilot vs full

Every command that touches data takes `--pilot` (default) or `--full`.
**Pilot = a fixed-seed 30% of the questions, stratified by task** (≈1,140 MVBench
questions, 150 LingoQA questions). The same seed always selects the same questions,
so pilot runs on different days are directly comparable.

---

## Quick start (repo)

```bash
git clone https://github.com/<you>/foveate && cd foveate
bash scripts/setup_gpu.sh          # picks a vLLM build that matches your NVIDIA driver
foveate env                        # GPU, driver, library versions

# Everything, end to end (pilot, both GPUs):
MODE=pilot bash scripts/run_all.sh
# ...then read results/REPORT.md and results/figures/
```

Data goes to `./data` (override with `FOVEATE_DATA`), results to `./results`
(`FOVEATE_RESULTS`), decoded-frame cache to `data/frame_cache` (`FOVEATE_CACHE`).

### Stage by stage

```bash
# Stage 1: batch-1 stage timing (synthetic frames: no dataset needed)
foveate profile --model 3b --source synthetic

# Stage 2: data + quality grid (one process per GPU with --gpus 2)
foveate download mvbench --tasks all          # ~17 GB; --tasks lite skips the 11 GB Perception zip
foveate grid --dataset mvbench --pilot --model 3b --gpus 2
foveate grid --dataset mvbench --pilot --model 7b --gpus 2

# Stage 4 data: rerun the ladder on fog / night / blur copies
foveate shift --dataset mvbench --pilot --model 3b --gpus 2
foveate shift --dataset mvbench --pilot --model 7b --gpus 2

# Stages 3-4: replay every policy from the cache (CPU, seconds)
foveate replay --dataset mvbench --pilot

# Driving demo
foveate download lingoqa
foveate grid  --dataset lingoqa --pilot --model 3b --gpus 2
foveate grid  --dataset lingoqa --pilot --model 7b --gpus 2
foveate judge --dataset lingoqa --pilot
foveate replay --dataset lingoqa --pilot

# Stage 5: vLLM servers (3B on GPU 0, 7B-AWQ on GPU 1) + gateway + load test
MODE=pilot bash scripts/serve_and_load.sh

# Figures and the summary
foveate report
```

Every grid run is **resumable**: if a Kaggle session dies, rerun the same command
and it skips questions already in the output files.

### Rough GPU time on Kaggle 2× T4

These are planning estimates from FLOP arithmetic, not measurements. Your Stage 1
profile replaces them with real numbers.

| Step | Pilot (30%) | Full |
|---|---|---|
| setup + MVBench download | 20–30 min | same |
| profile | ~15 min | same |
| grid, 3B (7 configs) | ~45 min | ~2.5 h |
| grid, 7B-AWQ (2 configs) | ~25 min | ~1.5 h |
| shift (3 corruptions × 3 configs) | ~60 min | ~3.5 h |
| LingoQA (both models + judge) | ~10 min | ~30 min |
| serving load tests | ~30 min | same |

Kaggle sessions stop after 12 hours and give 30 GPU-hours a week, so run the pilot
first, then the full grid, and the full shift study in a second session if needed.
Set `RUN_LINGOQA=0`, `RUN_SERVING=0` or `TASKS=lite` to trim.

---

## What you get

`results/REPORT.md` with tables and figures, all generated from result files:

- **Stage 1:** latency breakdown per (frames, resolution); TTFT vs visual tokens with a
  linear fit; prefill vs decode; achieved TFLOP/s vs the GPU's spec peak.
- **Stage 2–3:** accuracy (bootstrap 95% CI) vs cost for every static config, the
  learned rules, the cascade, rules+cascade and the oracle, on a held-out half, with
  paired accuracy differences; confidence AUROC/ECE per config; per-task heatmap;
  ablations.
- **Stage 4:** escalation rate and accuracy clean vs shifted; a clean→fog stream with
  a drift alarm; cost of a drift-aware cascade.
- **Stage 5:** throughput, p50/p95/p99 latency and TTFT for the cascade vs the best
  fixed config under the same Poisson trace.

## Theory, one note per stage

1. [Anatomy of VLM inference](theory/01_anatomy_of_vlm_inference.md): visual tokens, prefill vs decode, roofline, KV cache, timing correctly, T4 caveats
2. [Benchmarks, confidence and statistics](theory/02_benchmarks_confidence_and_statistics.md): MVBench, logit scoring, AUROC/ECE, pilot precision, paired bootstrap
3. [Adaptive compute and cascades](theory/03_adaptive_compute_and_cascades.md): the cost model, held-out thresholds, rules, oracle gap, related work
4. [Distribution shift and drift](theory/04_distribution_shift_and_drift.md): covariate shift, a free detector, why drift ≠ failure
5. [Serving VLMs under load](theory/05_serving_vlms_under_load.md): continuous batching, PagedAttention, prefix/encoder caches, queueing, goodput

## Repository layout

```
src/foveate/
  tokens.py            visual-token, FLOP and KV-cache arithmetic (+ GPU spec table)
  data/                MVBench, LingoQA, synthetic loaders; frame sampling/cache; corruptions
  backends/            hf_backend.py (stage timing via CUDA events), vllm_backend.py (fast path)
  profile.py           Stage 1 sweep          grid.py     Stage 2 runner (sharded, resumable)
  policy.py            Stage 3 policies       drift.py    Stage 4 shift + stream detector
  serve/               Stage 5 gateway (FastAPI) and Poisson load generator
  report.py            figures + REPORT.md    cli.py      `foveate ...`
  configs/default.yaml models, grids, ladder, shift and profile settings
theory/                one note per stage
notebooks/             the self-contained Kaggle notebook (built by tools/build_notebook.py)
tests/                 CPU tests, incl. a tiny random Qwen2.5-VL and a fake-model pipeline
```

To change what runs, edit `src/foveate/configs/default.yaml` or pass
`--config my.yaml`. Configs are written `MODEL:FRAMESxRES`, e.g. `3b:8x448`.

## Troubleshooting

- **vLLM will not install or start.** The grid's default `--backend auto` falls back to
  Transformers (batch 1, several times slower, same results format). The 7B rung needs
  vLLM because it is an AWQ checkpoint; without it, run the 3B grid only and the
  cascade uses the 3B rungs. You can force a vLLM version with
  `VLLM_OVERRIDE="vllm==0.11.0" bash scripts/setup_gpu.sh`.
- **NaN confidences or one repeated answer.** That is fp16 overflow on a GPU without
  bf16. Rerun with `--dtype float32 --tp 2` (3B only; it then spans both T4s).
- **Out of memory at start-up.** Add `--enforce-eager` and/or `--gpu-mem 0.85`.
- **LingoQA download fails.** Google Drive rate-limits shared folders. Download the
  evaluation folder manually from the link in the
  [LingoQA repo](https://github.com/wayveai/lingoqa) into `data/lingoqa/` and unzip it.
- **Disk.** MVBench needs ~17 GB after extraction (zips are deleted as they are
  extracted). Use `--tasks lite` (~6 GB) when space is tight.

## Honest limits

- MVBench accuracy at 4–16 frames will sit below Qwen's published 67.0 / 69.6, which
  used up to 768 frames. The comparison that matters is between policies on the same
  questions.
- One NTU-based MVBench task (`fine_grained_pose`) is not redistributable and is skipped.
- LingoQA scoring depends on a learned judge.
- Synthetic corruptions are a robustness study, not evidence of on-road safety. This is
  not a driving system.
- Novelty: cascades, routing and adaptive frame selection all have prior work (see
  theory/03). The contribution here is a measured, held-out cost/accuracy trade-off on a
  real serving stack.

## Licences

Code: MIT. MVBench is distributed under MIT on the Hugging Face Hub (its source
videos keep their original licences). LingoQA has its own licence; read it before
redistributing anything derived from it. Model weights follow the Qwen licences.
