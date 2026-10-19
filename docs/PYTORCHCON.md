# PyTorchCon 2026 notes

## The 30-second version

Video VLMs spend most of their compute on visual tokens. For Qwen2.5-VL the count is
`ceil(frames/2) × (res/28)²`: two knobs, plus model size. Foveate profiles where the
milliseconds go on T4s with CUDA events, caches every MVBench question under a grid of those
knobs in vLLM, and replays policies offline. The policies are fixed configs, keyword rules,
and a confidence cascade that escalates from 3B at 4 × 224 to 7B-AWQ at 16 × 448 only when
the option-letter probability is low. The threshold is fit on half the questions and
reported on the other half, with paired bootstrap CIs, a count of answers escalation broke,
and a non-deployable oracle as the ceiling. Then the same policy runs behind a gateway in
front of two vLLM servers, one per GPU, under Poisson load. The open question is whether VLM
confidence is calibrated enough for this to beat a good static config once the failed cheap
attempts are paid for.

## Three things worth discussing with vLLM / serving people

1. **Escalation re-pays the visual prefill.** A cascade that moves from 4 to 16 frames
   re-encodes the clip. Prefix caching does not help, because the visual tokens differ. A
   multimodal encoder cache helps only for identical frames. Is there a sane way to reuse
   the ViT work across frame counts, for example by encoding at the higher rung's frame
   indices and subsampling?
2. **Per-request confidence as a scheduling signal.** The cascade only needs the first
   decode step's logits over a few option tokens. vLLM's restricted logprobs make that
   cheap. Could a router act on it *inside* the engine, preempting and re-queuing at a
   higher rung, instead of a second HTTP round trip?
3. **T4-class hardware.** No bf16, no FlashAttention-2, 320 GB/s memory bandwidth. Decode
   is bandwidth-bound (~19 ms/token floor for 3B fp16 by the weights/bandwidth bound, a
   theoretical number) while the ViT and prefill are compute-bound. Which kernels does vLLM
   actually pick on sm75, and what is the achieved MFU on the ViT? Stage 1 measures the
   latter.

## Likely questions, and honest answers

- *Does the cascade win?* Not known yet: no GPU run has been made. The harness reports it
  either way, including "not significantly different" verdicts and answers broken by
  escalation.
- *Isn't confidence from a VLM badly calibrated?* Possibly. That is exactly what the AUROC
  and ECE per config, plus the overconfident-failure list, are there to show. If AUROC is
  near 0.5 the cascade cannot work, and the report will say so.
- *Why MVBench?* It is in nearly every recent video-VLM paper, every answer is one letter
  (cheap, exact scoring from log-probs), and published Qwen2.5-VL numbers exist as a sanity
  anchor. Our absolute accuracy will be lower, because we use far fewer frames.
- *Why not just use a router model?* That is related work (theory/03). The rules policy is
  a cheap stand-in. The point of the comparison is cost-aware and held-out, not novelty of
  the router.
- *Is this production ready?* No. It is a measured research harness. The serving stage is
  a load test against vLLM, not a hardened service.

## What to have on screen

- `docs/architecture.svg`
- `results/smoke/SMOKE.md` from a real Kaggle run (proves the stack runs)
- After a pilot: `results/figures/frontier_mvbench_pilot.png` and `profile_breakdown.png`,
  both generated from real result files by `foveate report`
