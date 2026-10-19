# Run modes and what each one writes

Every data command takes one of three modes. They select **which questions** and, for
smoke, **which configs**. Everything else (prompts, scoring, file formats) is identical, so
a smoke run exercises exactly the code a full run uses.

| | smoke | pilot | full |
|---|---|---|---|
| Flag | `--smoke` / `--subset smoke` | `--pilot` (default) | `--full` |
| Questions | 8: drawn from the pilot set, round-robin over tasks | fixed-seed 30% per task (seed 0) | all |
| Configs | `smoke.configs` in `default.yaml` (MVBench: 3B `4x224`, `8x448`; 7B `8x448`) | `grid` | `grid` |
| Results dir | `results/<dataset>_smoke/` | `results/<dataset>_pilot/` | `results/<dataset>_full/` |
| Resumable | yes | yes | yes (re-run the same command; finished question IDs are skipped) |
| Runtime | measured by the run itself; no estimate is given | ″ | ″ |

The sets are nested: smoke ⊂ pilot ⊂ full. The same seed always selects the same questions,
so runs on different days are comparable.

## Smoke: `foveate smoke --models 3b 7b --profile`

Uses the generated synthetic dataset (12 clips encoded as MP4, 24 questions) unless you
pass `--dataset mvbench` after `foveate download mvbench --tasks <one task>`. It needs no
download, and the frame cache is private, so video decoding really runs.

What runs:

1. Environment: GPU visible, compute capability, the dtype that will be used (float16 on a T4), and package versions.
2. Per model, in its own process (3B on GPU 0, 7B on GPU 1 when two GPUs exist):
   `foveate grid --subset smoke --probe 4 --batch-size 8`.
3. Validation of what that wrote (below).
4. With `--profile`: one CUDA-event-timed setting (3B, 4 frames, 224 px, 8 output tokens,
   1 warm-up, 3 timed runs) plus the cold first call.

Expected outputs:

```
results/smoke/SMOKE.md                 PASS/WARN/FAIL table, one row per check (the verdict)
results/smoke/SMOKE.json               same, plus env, question IDs, per-config costs
results/smoke/grid_3b.log, grid_7b.log full logs of the grid processes
results/smoke/profile_3b.jsonl         (--profile) raw timings, incl. a "cold" row
results/synthetic_smoke/grid/3b_2x224.s0of1.jsonl   one row per question (8 rows)
results/synthetic_smoke/grid/3b_8x448.s0of1.jsonl
results/synthetic_smoke/grid/7b_8x448.s0of1.jsonl
results/synthetic_smoke/cost/<same stems>.json      backend, engine start (cold), first batch,
                                                    amortized ms, warm batch-1 p50/p95/p99
results/synthetic_smoke/skipped.jsonl  only if a config did not fit (with the reason)
results/synthetic_smoke/events.jsonl   only if vLLM failed and the run fell back to Transformers
```

A healthy smoke run has no FAIL. It also has:
- backend `vllm` for every model (a WARN means it fell back to Transformers, so timings are not vLLM's);
- `visual tokens = formula` PASS, which means the engine's token count matches `tokens.py`;
- no NaN confidences, confidences in [0, 1], predictions that are option letters, answers that vary;
- synthetic accuracy above 0.5 (WARN level only: these are easy generated questions);
- positive warm batch-1 latency and a recorded engine start time;
- with `--profile`: positive `vision_ms`, `prefill_ms` and `decode_ms`.

Exit code 1 if any check FAILs. The notebook stops there.

## Pilot and full

`MODE=pilot GPUS=2 bash scripts/run_all.sh` (or `full`) runs smoke first and stops if it
fails. It then runs every stage. Expected outputs:

```
results/env.json                                   foveate env (GPU, driver, versions)
results/profile/3b_synthetic.jsonl                 stage 1 raw timings (incl. cold row, skipped settings)
results/profile/3b_synthetic.summary.json          medians, p95, MFU per (frames, res, out_tokens)
results/mvbench_<mode>/grid/*.jsonl                stage 2 rows, one file per config per GPU shard
results/mvbench_<mode>/grid/*__fog3.*.jsonl        stage 4 shifted rows (fog/night/blur at severity 3)
results/mvbench_<mode>/cost/*.json                 per config per shard: backend, cold/first/steady, latency
results/mvbench_<mode>/skipped.jsonl               configs that were not run, with reasons (if any)
results/mvbench_<mode>/replay/summary.json         stage 3: policies, baselines, verdicts, escalation, curve
results/mvbench_<mode>/replay/policy.json          the fitted ladder + tau + rules the gateway serves
results/mvbench_<mode>/replay/shift.json           stage 4: per-corruption escalation, drift streams
results/lingoqa_<mode>/...                         same layout for the driving demo (after `judge`)
results/serving/vllm_3b.log, vllm_7b.log, gateway.log
results/serving/gpu_placement.csv                  which process holds memory on which GPU
results/serving/gateway_health.json                routing check: model served by each backend
results/serving/<policy>_r<rate>.jsonl             one record per request (rungs, ttft, latency)
results/serving/<policy>_r<rate>.summary.json      throughput, p50/p95/p99, cold_*, accuracy, escalation
results/serving/telemetry.jsonl                    gateway-side record of every rung of every request
results/REPORT.md, results/figures/*.png           foveate report
```

The pilot keeps every task's share, so per-task accuracy is still estimable. The full run
narrows the confidence intervals. The policy comparison method is the same in both.
