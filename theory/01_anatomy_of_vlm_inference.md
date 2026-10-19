# 01 · Anatomy of VLM inference (Stage 1: the profiler)

> **One sentence to remember:** for a short answer about a video, the cost is set
> almost entirely by how many visual tokens you create, and you create them with two
> knobs: how many frames you send and how many pixels each frame has.

This note explains what `foveate profile` measures and why the numbers come out the
way they do. All arithmetic uses Qwen2.5-VL-3B on a Kaggle **T4** (16 GB, fp16).
Where a number is a back-of-envelope estimate, it says so. Your profiler output is
the ground truth that confirms or corrects it.

---

## 1. The life of one request

```
 video file ──► decode + sample frames ──► resize ──► processor (patchify, normalise)     CPU
                                                         │
                                       H2D copy ◄────────┘
                                          │
                  ┌───────────────────────▼──────────────────────┐
                  │ vision encoder (ViT, ~0.67B params)           │  GPU, runs once
                  │ 14×14 patches → features → 2×2 merge → tokens │
                  └───────────────────────┬──────────────────────┘
                                          ▼
                  ┌──────────────────────────────────────────────┐
                  │ LLM prefill: all visual + text tokens at once │  GPU, compute-bound
                  │ writes the KV cache  ──────────► TTFT ends     │
                  └───────────────────────┬──────────────────────┘
                                          ▼
                  ┌──────────────────────────────────────────────┐
                  │ decode: 1 token per step, reads all weights   │  GPU, memory-bound
                  │ + KV cache every step ──────────► TPOT         │
                  └──────────────────────────────────────────────┘
```

The profiler times each box separately (see `src/foveate/backends/hf_backend.py`):

| Stage | How it is timed |
|---|---|
| load (decode + resize) | `time.perf_counter` around `FrameStore.load` |
| preprocess | `perf_counter` around the HF processor call |
| H2D | CUDA events around `.to(device)` |
| vision encoder | CUDA events from forward **pre/post hooks** on `model.visual` |
| prefill | first generate step minus H2D minus vision |
| decode | events from a do-nothing **logits processor**, which runs once per generated token |

---

## 2. Where visual tokens come from

Qwen2.5-VL cuts each frame into **14×14-pixel patches**, then an MLP **merges every
2×2 block of patches into one LLM token**. So one token covers **28×28 pixels**. For
video it also fuses **every 2 consecutive frames** into one temporal patch (a 3D
convolution with temporal size 2), which halves the token count again.

$$
\text{visual tokens} = \left\lceil \frac{F}{2} \right\rceil \cdot \frac{H}{28} \cdot \frac{W}{28}
$$

| Setting | Visual tokens |
|---|---|
| 4 frames, 224×224 | 2 × 64 = **128** |
| 8 frames, 448×448 | 4 × 256 = **1,024** |
| 16 frames, 448×448 | 8 × 256 = **2,048** |
| 8 frames, 672×672 | 4 × 576 = **2,304** |
| 16 frames, 672×672 | 8 × 576 = **4,608** |

**Doubling resolution (per side) multiplies tokens by 4; doubling frames multiplies
them by 2.** Resolution is the expensive knob. `tokens.py` implements the exact
rounding rule (`smart_resize`), and `tests/test_tokens.py` checks the table above.

> **Trend check.** Every frontier API now exposes this knob to users. OpenAI's image
> input has a `detail: low` mode that costs a fixed 85 tokens; Gemini has a
> `media_resolution` setting. Inside models, token reduction is an active research
> line: FastV (ECCV 2024) drops low-attention visual tokens after layer 2, and
> VisionZip and PyramidDrop (both CVPR 2025) prune or merge tokens. NVIDIA's NVILA
> calls the recipe "scale then compress". Foveate works *outside* the model (frames
> and pixels), so it composes with any of those.

---

## 3. Prefill is compute-bound

A transformer forward pass costs about **2 FLOPs per parameter per token** for the
matrix multiplies. Prefill processes all prompt tokens in parallel, so the GPU's
tensor cores stay busy.

For 16 frames at 448 (2,048 visual tokens + ~100 text tokens):

- **LLM:** 2 × 3.09e9 × 2,150 ≈ **13.3 TFLOP**
- **Vision encoder:** it runs on the patches *before* the 2×2 merge, so 4× as many:
  2 × 0.67e9 × 8,192 ≈ **11.0 TFLOP**

A T4 peaks at **65 TFLOP/s** in fp16 on paper. Real kernels reach roughly 30 to 50%
of that, so this request needs about **0.75 to 1.2 s** of prefill+vision on a T4.

**The non-obvious finding to look for:** on a 3B VLM, the vision encoder is nearly
**half** of the prefill compute (11.0 vs 13.3 TFLOP), because it sees 4× more
positions than the LLM. On the 7B model the LLM is 2.5× bigger while the ViT is the
same, so the vision share drops to about a quarter. Small VLMs are vision-bound
earlier than people expect. The profiler's `vision_mfu` and `prefill_mfu` columns
(achieved TFLOP/s ÷ spec peak) let you check this.

---

## 4. Decode is memory-bound

Decode generates **one token per step**. Each step multiplies a single vector by every
weight matrix, so the GPU does little math per byte loaded. The limit is how fast it
can stream the weights from memory.

$$
\text{TPOT}_{\min} \approx \frac{\text{weight bytes}}{\text{memory bandwidth}}
= \frac{3.09\text{e}9 \times 2\ \text{B}}{320\ \text{GB/s}} \approx 19\ \text{ms}
$$

That is a ceiling of about 50 tokens/s for one request on a T4. **Batching** is how
servers beat it: 32 requests in one step share a single read of the weights. This is
the whole reason continuous batching exists (see note 05).

For multiple-choice questions Foveate generates **one token**, so decode is almost
free and the question becomes purely a prefill question. The profiler's decode sweep
(1, 16, 64, 256 output tokens) shows the crossover point where decode starts to
dominate.

### Roofline in one line

Arithmetic intensity = FLOPs per byte moved. The T4's ridge point is
65e12 / 320e9 ≈ **203 FLOP/byte**. Prefill on thousands of tokens sits far above it
(compute-bound); batch-1 decode sits near **1 FLOP/byte** (memory-bound).

---

## 5. KV cache

Each token stores a key and a value vector per layer per KV head:

$$
\text{KV bytes/token} = 2 \times L \times H_{kv} \times d_{head} \times 2\ \text{B}
$$

Qwen2.5-VL-3B: 2 × 36 × 2 × 128 × 2 = **36 KB per token** (grouped-query attention
with only 2 KV heads keeps it small). A 2,048-token video request needs about 75 MB.
KV is not the bottleneck at batch 1. It *is* the bottleneck when vLLM packs dozens of
video requests into one T4, which is when PagedAttention matters (note 05).

---

## 6. Measuring correctly (the part most benchmarks get wrong)

1. **CUDA is asynchronous.** A kernel launch returns before the GPU finishes, so
   `time.time()` around a model call measures Python, not the GPU. Use
   `torch.cuda.Event(enable_timing=True)`, which is recorded inside the GPU stream, and
   synchronize once *after* the timed region.
2. **Warm up.** The first calls pay for CUDA context creation, cuBLAS algorithm
   selection and the caching allocator growing. The profiler discards warm-up runs.
3. **Report medians and p95, not means.** Latency distributions have right tails.
4. **Record the environment.** GPU, driver, CUDA, torch, transformers and vLLM
   versions go in `*.env.json` next to every result.
5. **Say where each number came from.** Stage timings come from Transformers
   (because we own the model object). Serving numbers (TTFT under load, p95) come from
   vLLM. They are different engines and will not match exactly.

---

## 7. T4-specific caveats (Kaggle)

- **No bf16.** Turing GPUs (compute capability 7.5) support fp16 and fp32 only.
  Qwen2.5-VL was trained in bf16, and fp16 has a much smaller range (max ≈ 65,504), so
  activations can overflow. The grid runner checks the first batch for NaNs or a single
  repeated answer. If you see that, rerun with `--dtype float32 --tp 2` (the 3B model
  in fp32 needs ~15 GB of weights, which fits only split across both T4s).
- **No FlashAttention-2** (it needs Ampere or newer). Transformers uses PyTorch SDPA,
  and vLLM picks its Triton attention backend on Turing.
- **Two GPUs, used as data parallel.** `--gpus 2` runs one process per T4, each on half
  the questions. That beats tensor parallelism here because the T4s talk over PCIe,
  and a 3B model fits on one card anyway.

---

## 8. What to look for in your Stage 1 results

- `ttft_vs_tokens.png`: TTFT should be close to **linear in visual tokens** (R² near 1).
  The slope is your GPU's "ms per visual token". A curve bending upward means attention's
  quadratic term is starting to show.
- `profile_breakdown.png`: at low token counts, CPU **load + preprocess** can rival the
  GPU time. That is a real finding: on Kaggle's 4 vCPUs the video decoder can become the
  bottleneck, which is why Foveate caches decoded frames.
- `prefill_mfu`: if it is far below 0.3, something is inefficient (padding, small
  kernels, fp32 fallback).

**Read next:** the Qwen2.5-VL technical report (arXiv 2502.13923), sections on the
vision encoder and dynamic resolution; the PyTorch profiler docs; then note 02.

---

## 9. Measured results (pilot run, Kaggle 2x T4, Oct 19 2026)

From `results/REPORT.md`, Stage 1 (Qwen2.5-VL-3B, Transformers, batch 1, CUDA events,
medians after warm-up). GPU time below = vision encoder + prefill.

| Setting | Visual tokens | Vision (ms) | Prefill (ms) | GPU time (ms) | CPU load + preprocess (ms) |
|---|---|---|---|---|---|
| 1 × 224 | 60 | 42 | 72 | 114 | 16 |
| 4 × 448 | 504 | 262 | 310 | 572 | 71 |
| 8 × 448 | 1,008 | 511 | 627 | 1,138 | 161 |
| 1 × 896 | 1,008 | 544 | 637 | 1,181 | 58 |
| 16 × 448 | 2,016 | 1,024 | 1,710 | 2,734 | 327 |
| 16 × 672 | 4,464 | 2,378 | 5,858 | 8,236 | 482 |

What the predictions in this note got right and wrong:

- **Tokens set the cost (confirmed).** GPU time is about 1.2 ms per visual token up to
  ~1,100 tokens (fit R² 0.99 up to 2,000 tokens), whatever produced the tokens: 1 frame
  at 896 px and 8 frames at 448 px both give 1,008 tokens and cost 1,181 vs 1,138 ms.
- **Superlinear past ~2,000 tokens.** 1.85 ms per token at 4,464 tokens: attention's
  quadratic term and memory pressure. 16 × 896 ran out of memory on the 16 GB card.
- **The vision encoder is 37 to 48% of GPU time (confirmed "nearly half").**
- **Section 3's estimate was optimistic.** 16 × 448 took 2.7 s of GPU time against the
  0.75 to 1.2 s estimate: measured `prefill_mfu` was 0.07 to 0.18, below the 30 to 50%
  assumed. Batch-1 Transformers kernels on a T4 leave most of the peak unused.
- **Decode.** TPOT was ~51 ms against the 19 ms bandwidth floor (about 37% of it), in
  Transformers at batch 1 without CUDA graphs.
- **Cold start.** The first request took 3,697 ms of GPU time at 1 × 224 against a
  1,698 ms warm median (both with 32 decode steps).
- **CPU matters.** Load + preprocess reached 482 ms at 16 × 672; on vLLM, video decode
  was ~340 to 370 ms per question in the smoke test, larger than the model call.
