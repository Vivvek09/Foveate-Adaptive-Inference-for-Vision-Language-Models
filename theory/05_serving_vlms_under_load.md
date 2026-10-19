# 05 · Serving VLMs under load (Stage 5: the gateway and load test)

> **One sentence to remember:** under load, p95 latency is mostly *waiting*, so a
> policy that makes the average request cheaper also shortens everyone's queue.

---

## 1. The system on Kaggle's two T4s

```
 load generator ──HTTP──► foveate gateway (CPU) ──► vllm serve Qwen2.5-VL-3B      GPU 0, :8001
 (Poisson arrivals)        frames, resize, cascade └► vllm serve Qwen2.5-VL-7B-AWQ GPU 1, :8002
                           telemetry JSONL
```

The split of responsibilities is deliberate:

- **vLLM** owns everything per-token: continuous batching, PagedAttention, prefix
  caching, the multimodal encoder cache.
- **The gateway** owns everything per-request that vLLM does not know about: how many
  frames, which resolution, which model, when to escalate.

Placing one model per GPU turns "route to the bigger model" into "route to the other
GPU", which is how multi-model serving fleets work.

---

## 2. What vLLM does for you (and why each piece exists)

**Continuous batching** (Orca, OSDI 2022). A naive server batches requests and waits
for the slowest to finish. Continuous batching re-forms the batch at *every decode
step*: finished sequences leave, new ones join. Since batch-1 decode is memory-bound
(note 01), adding requests to a step is nearly free until compute saturates.

**PagedAttention** (vLLM, SOSP 2023). KV cache is stored in fixed-size blocks, like
virtual-memory pages, so memory is not reserved for each request's maximum length.
This roughly doubled to quadrupled batch sizes in the paper, and batch size is
throughput.

**Chunked prefill** (Sarathi-Serve, OSDI 2024). A 4,600-token video prefill would stall
every other request's decode. Splitting prefill into chunks and mixing them with
decode steps keeps time-between-tokens smooth for everyone.

**Prefix caching** (also SGLang's RadixAttention, 2024). If two requests share a
prompt prefix, the KV cache for that prefix is reused. Foveate's prompt puts the
**video before the question**, so "same clip, different question" requests share the
entire visual prefix. For multimodal inputs vLLM hashes the media content, so
identical frames hit the cache. The load generator's `--repeat` knob controls how
often this happens.

**Encoder cache / processor cache.** vLLM also caches vision-encoder outputs and
preprocessed media between requests, so a repeated clip can skip the ViT, the second
biggest cost from note 01.

> **Trend check.** The frontier of serving research is *disaggregation*: run prefill
> and decode on different GPUs (DistServe and Splitwise, both 2024), and for
> multimodal models, split the encoder off too ("encode-prefill-decode", EPD,
> 2024-25). The vLLM sessions at PyTorch Conference 2026 were about KV-cache transfer
> and offloading for exactly this. Foveate's per-stage profile (note 01) is the
> measurement you would need to decide whether encoder disaggregation pays off.

---

## 3. Queueing: why p95 explodes

Model a server as a queue with arrival rate λ and service rate μ (requests per
second it can finish). Utilisation is ρ = λ/μ. For the simplest queue (M/M/1), mean
time in system is

$$
T = \frac{1}{\mu - \lambda} = \frac{1/\mu}{1 - \rho}
$$

At ρ = 0.5 requests take 2× their service time; at ρ = 0.9, 10×. Tails are worse than
means. Two consequences for this project:

1. **A cheaper average request lowers ρ for everyone.** If the cascade cuts mean
   service cost by 35%, μ rises, ρ falls, and the p95 improvement under load can be
   much larger than 35%. This is the "under load" result Stage 5 exists to show.
2. **Escalations add variance.** A cascade request that escalates pays twice. Mean
   cost falls but the cost distribution gets a fatter right tail, which pushes p99
   up. Report p50, p95 *and* p99.

**Little's law** (L = λ·W) is the sanity check: the average number of requests in
flight equals arrival rate times average latency. If your telemetry violates it,
something is mismeasured.

---

## 4. Running a fair load test

- **Open loop, Poisson arrivals.** A closed-loop client (send, wait, send) slows down
  when the server slows down, hiding the queue. `foveate loadtest` schedules every
  arrival in advance from a seeded exponential distribution.
- **Same trace for every policy.** Static and cascade policies see the identical
  sequence of questions at identical times.
- **Sweep the rate.** One rate tells you little. Run a low rate (latency is pure
  service time), a medium one, and one near saturation, and plot p95 against rate for
  both policies. The cascade's curve should bend upward later.
- **Goodput.** DistServe defines goodput as requests per second that *meet* a latency
  SLO (for example p95 TTFT < 2 s). It is a better headline than raw throughput,
  because throughput can rise while every user waits longer.
- **Warm the servers** with a few requests first, and keep the gateway's frame cache
  warm too, or the first minute measures disk reads.

---

## 5. T4 serving notes

- The 3B model in fp16 is ~7.5 GB of weights, leaving ~6 GB of KV cache on a 16 GB T4
  at `--gpu-memory-utilization 0.9`. At 36 KB/token that is ~170k tokens, about 80
  concurrent 2,048-token video requests in theory, fewer in practice because of
  activation memory and the encoder cache.
- The 7B-AWQ model is ~6 GB of int4 weights; its KV is 56 KB/token (28 layers, 4 KV
  heads).
- `--enforce-eager` skips CUDA-graph capture: faster start-up and less memory, slower
  decode. Use it if start-up fails on memory.
- Kaggle notebooks can reach `localhost` ports, so the whole stack runs inside one
  notebook with servers in background processes.

---

## 6. What a good Stage 5 result looks like

A table from `REPORT.md` with, for the same Poisson trace at two or three rates:
throughput, p50/p95/p99 latency, p95 TTFT, accuracy and escalation rate for "always
16×448 on 3B" versus the cascade. The claim you can defend is the one the table
supports, including the honest case where escalation variance hurts p99.

**Read next:** the vLLM paper (arXiv 2309.06180), sections 1 to 4; Sarathi-Serve
(arXiv 2403.02310); DistServe (arXiv 2401.09670), especially its definition of goodput.

---

## 7. Measured results (serving run, Kaggle 2x T4, Oct 19 2026)

From `results/serving/*.summary.json`: gateway + vLLM 0.32 (3B fp16 on GPU 0,
7B-AWQ on GPU 1), the pilot's tuned cascade (τ = 0.981) against static 7b:16x448
(the replay's reference). 200 seeded Poisson requests per run, 30% repeated clips,
5 warm-up requests excluded; 200 of 200 succeeded in every run.

| Rate | Policy | Throughput (req/s) | p50 (s) | p95 (s) | p99 (s) | ≤ 10 s | Accuracy | Escalated |
|---|---|---|---|---|---|---|---|---|
| 0.5 | cascade | 0.43 | 8.2 | 20.4 | 23.7 | 59% | 0.685 | 86.5% |
| 0.5 | static 7b:16x448 | 0.43 | 16.5 | 30.5 | 33.9 | 28% | 0.685 | 0% |
| 1 | cascade | 0.54 | 64.6 | 135.2 | 149.1 | 7% | 0.685 | 86.5% |
| 1 | static 7b:16x448 | 0.42 | 86.9 | 245.9 | 257.3 | 4% | 0.685 | 0% |
| 2 | cascade | 0.54 | 103.2 | 237.7 | 257.8 | 4% | 0.685 | 86.5% |
| 2 | static 7b:16x448 | 0.42 | 134.6 | 344.4 | 365.7 | 1% | 0.685 | 0% |

- **Section 3's prediction held, for a different reason than expected.** The cascade does
  *more* total GPU work per request than the static 7B (note 03, section 8), yet it
  halved p50 at 0.5 req/s and doubled goodput (share within 10 s: 59% vs 28%).
- **Why (interpretation, consistent with the queueing model):** measured capacity is
  ~0.42 req/s for static and ~0.54 req/s for the cascade (throughput when overloaded).
  Static sends everything to GPU 1 and leaves GPU 0 idle; at the 0.46 req/s actually
  offered it runs at ρ ≈ 1.1 and its queue grows. The cascade runs its 3B rungs on
  GPU 0 and sends only 71% of requests to GPU 1 (ρ ≈ 0.86). Under load the busiest
  GPU's utilisation is the cost that matters, not summed milliseconds.
- **Saturation.** Two T4s serve about 0.4 to 0.5 req/s for 16-frame video questions.
  At 1 and 2 req/s latency is almost all queueing (a single request on idle servers
  took 3.1 to 3.7 s).
- **The live path is slower than the grid.** One warm 7b:16x448 request took 1.6 s in
  the grid but ~3.7 s through the gateway; video decode, JPEG + base64 transfer and
  multimodal preprocessing are the likely extra costs (not separately measured).
- **vLLM 0.32 needs `--trust-request-mm-kwargs`** for the gateway's per-request fps
  (`media_io_kwargs`); without it every request fails with a 400.
