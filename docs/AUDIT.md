# Release audit (v0.1.0, 2026-10-19)

This is what was checked before the first public push, what was changed, and what is still
unverified. "CPU" means verified by the test suite on CPU. "Mocked" means verified against a
stand-in model or a fake server. "GPU" means run on real hardware. **Nothing in this release
has run on a GPU.**

## Findings and fixes by stage

### 1 · Profile
| Finding | Fix |
|---|---|
| Warm-up runs were discarded, but the cold first call was never reported | The first call of the process is now a `sweep: "cold"` row, shown next to the warm median in the report |
| Settings that cannot fit (e.g. 16 frames × 896 px) were attempted, then caught as OOM | Memory/context guard before each setting; skipped settings are recorded with a reason |
| CUDA events measure GPU-timeline time between marks, which includes host-induced idle gaps (e.g. rope-index computation) | Documented: "prefill" = TTFT − H2D − vision on the GPU timeline, not pure kernel time |

### 2 · Benchmark grid
| Finding | Fix |
|---|---|
| `dtype="auto"` was resolved per backend; an explicit bf16 on T4 would silently run emulated | `resolve_dtype`: auto → fp16 below compute capability 8.0; explicit bf16 on a T4 raises |
| A vLLM start-up failure fell back to Transformers with only a log line; rows and costs did not say which engine produced them | Every row and cost file carries `backend` (`vllm`/`hf`/`fake`), the fallback is logged in `events.jsonl`, and a grid file refuses rows from a second backend |
| The 7B rung is an AWQ checkpoint; the HF fallback would have tried and failed (or needed another model) | On the HF path, AWQ configs are skipped with a recorded reason, never substituted |
| No pre-flight memory check; one T4 is a separate 15 GiB pool | `memory.check_fit`: weights + KV for one full sequence + activations + overhead, plus an exact context-length check, per GPU; misfits are written to `skipped.jsonl` |
| HF latency probe excluded the processor while the vLLM probe included it | Both now time processor + engine |
| Cold start, first batch and steady state were mixed into one number | Cost files record `backend_load_s` (cold), `first_batch_wall_s`, `steady_amortized_ms`, and warm batch-1 p50/p95/p99 after discarded warm-up calls |
| Mocked timings could look like measurements | `measured: true` only for real engines; the report marks other rows MOCKED and the smoke checker fails them |
| The fp16 advice for NaN said `--dtype float32`, which does not fit a 3B on one T4 | Now `--dtype float32 --tp 2` |

### 3 · Optimize (policy replay)
| Finding | Fix |
|---|---|
| The reference config was chosen as the most accurate on **all** questions, test half included (mild selection leakage) | Chosen on the calibration half only |
| τ candidates were quantiles of all questions' confidences (no labels, but test data) | Quantiles of calibration confidences only |
| The trade-off curve was drawn over all questions while the policy points were held-out | The curve is now drawn over the held-out half too |
| If one config lacked a measured latency, its TFLOPs were mixed with other configs' milliseconds | Units are never mixed: if any config lacks a real measurement, or configs were timed on different backends, every cost switches to model FLOPs with a note |
| Baselines were unnamed static configs | Named deployable baselines (fixed cheapest, static intermediate, fixed strongest); the oracle is labelled "not deployable" and excluded from the frontier |
| No analysis of what escalation did | Per policy: escalated / helped / hurt / no-change counts, accuracy with vs without escalation, extra cost, overconfident wrong answers (with the 10 most confident examples), and paired-bootstrap verdicts against each baseline |

Checked and correct: routing reads only confidences and question text (never labels). The
calibration/test split is stratified and hash-seeded. The paired bootstrap resamples the
same questions for both policies. Cascade cost is the sum of every rung run.

### 4 · Shift evaluation
No defects found. One note: τ is fit on clean calibration data and applied to all shifted
questions. That is intended (the policy does not know about the shift), but the shifted
numbers are not a held-out-split estimate.

### 5 · Serve
| Finding | Fix |
|---|---|
| The gateway trusted its backend map; a server serving the wrong model would go unnoticed | Startup check of `/v1/models` per backend; `/health` reports `routing_ok` |
| Static policies on a model without a server would error deep in a request | HTTP 400 with the reason; ladder rungs without a backend are dropped and logged |
| On a single GPU the script started both servers at 42% memory each, which cannot hold the 3B weights | One GPU → 3B server only; 7B rungs dropped (logged); the static 7B comparison is skipped with a reason |
| `RUN_7B=False` still started the 7B server | `SERVE_7B` follows `RUN_7B` |
| Policies ran back to back on warm servers, so the second inherited the first's prefix cache | Prefix cache reset between runs (vLLM dev endpoint, best effort), policy order alternates per rate, warm-up requests are sent untimed and reported as `cold_*` |
| No record of where each model actually ran | `gpu_placement.csv` from `nvidia-smi`; every rung's `backend_url` is in the telemetry |

Checked and correct: arrivals are open loop and seeded, every policy at a rate gets the
identical trace, and latency is measured at the client. Throughput is completed requests
divided by makespan. Percentiles use linear interpolation over successful requests only,
with the failure count reported.

## Kaggle T4 x2 compatibility (by inspection, not run)

| Component | Status |
|---|---|
| Precision | fp16 everywhere (T4 = sm75, no bf16); AWQ fp16 compute; `vllm serve --dtype half` |
| vLLM | build chosen from the driver's CUDA version; AWQ kernels need sm75+, so they run on a T4. FlashAttention-2 needs sm80, so vLLM picks a Turing-capable attention backend (recorded via `VLLM_ATTENTION_BACKEND` if you set it) |
| Transformers | `attn_implementation="sdpa"` (no FA2 on T4); the Qwen2.5-VL processor is called with no re-resize and no re-sampling |
| Memory | 3B fp16 ≈ 7.0 GiB weights, 7B-AWQ ≈ 6.6 GiB; every default grid config fits one T4 by the guard's estimate; 16 × 896 exceeds 8,192 tokens and is skipped in vLLM |
| Numerics | fp16 overflow is detected (NaN / constant answers) in the grid and in smoke, not prevented |

## Still unverified (do these first on a GPU)

1. `foveate smoke --models 3b 7b --profile` on Kaggle T4 x2: vLLM start-up, the (frames, metadata) video input path, exact letter log-probs, token count = formula, fp16 numerics.
2. `vllm serve` flags (`--limit-mm-per-prompt` JSON, `--mm-processor-kwargs`, `data:video/jpeg` URLs, `media_io_kwargs`) against the installed vLLM version.
3. MVBench zip layouts and the LingoQA parquet schema against the real downloads (the loaders have fallbacks, untested on real files).
4. Every number in REPORT.md: none exists yet.
