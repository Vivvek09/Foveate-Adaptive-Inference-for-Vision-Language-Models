# Foveate as a research project

This page states the research question, the methods, the evaluation protocol and what is
still open. The README reports what has been measured; this page says what the project is
trying to find out and how each claim would be tested.

**Status key:** *measured* = run on GPU and reported in the README. *implemented* = in the
code and unit-tested on CPU, GPU numbers not yet produced. *planned* = not written yet.

## Problem

A video VLM request has three compute knobs: frames, resolution and model size. Visual
tokens set most of the prefill cost, and the right setting depends on the question. A
serving system that picks one setting for every request either overpays on easy questions or
underserves hard ones. The systems question is:

> How much of the gap between a fixed setting and a per-question oracle can a deployable
> policy close, and does a saving measured offline survive a real serving stack under load?

The first MVBench run (October 2026) gave the size of the prize and the obstacle. The oracle
reaches 0.836 accuracy at 0.16x the reference cost, against 0.683 at 1.00x for the strongest
fixed setting. The confidence cascade does not collect it: cheap-rung confidence separates
right from wrong answers only moderately (AUROC about 0.71), so a safe threshold escalates
93% of questions, and a ladder pays for every rung it climbs.

## Research questions

| | Question | Status |
|---|---|---|
| RQ1 | Where does GPU time go as visual tokens grow? | measured (E1) |
| RQ2 | Can per-request allocation match the strongest fixed setting at lower cost on held-out questions? | measured for rules and cascade (E3); implemented for the router |
| RQ3 | Is confidence miscalibration the reason the cascade over-escalates, and does calibration fix it? | implemented (temperature scaling) |
| RQ4 | Does a policy's saving hold under distribution shift? | measured (E4) |
| RQ5 | Under load, which policy carries the most traffic within a latency target? | measured as p50/p95 (E5); implemented as goodput |
| RQ6 | Are the conclusions stable across calibration/test splits? | implemented (split robustness) |

## Methods added after the first run

### Probe-then-route (`src/foveate/router.py`)

The cascade climbs a ladder and pays for each rung. The router runs the cheapest rung once
as a **probe**, then makes one jump: it picks the config with the highest predicted accuracy
minus λ times its extra cost. A question therefore costs at most two runs.

- **Predictor.** One ridge-penalised logistic regression per config, estimating the
  probability that the config answers correctly. Inputs are only what is known after the
  probe: the probe's confidence, its logit, and the question's keyword category. No labels,
  no features from configs that were not run.
- **Choosing λ.** λ is the cheapest value whose accuracy on the calibration half reaches the
  reference config's. That accuracy is computed from **cross-fitted** predictions (5 folds),
  so no question is scored by a model that trained on it (Chernozhukov et al., 2018).
- **Relation to prior work.** Cascades that stop on a score go back to NoScope (Kang et al.,
  2017) for video and FrugalGPT (Chen et al., 2023) for LLMs; RouteLLM (Ong et al., 2024)
  learns a router between two LLMs from preference data. Certaindex (Fu et al., 2024) uses a
  certainty signal to decide how much more compute a reasoning query should get. Foveate's
  router applies the same idea to a different resource: visual tokens and model size for
  video questions, with the probe's confidence as the certainty signal.

### Temperature scaling (`router.py`)

The 7B-AWQ model is overconfident (ECE about 0.2) while the 3B model is not (about 0.05), yet
the cascade applies one threshold to both. One temperature per config is fit on the
calibration half (Guo et al., 2017) and the threshold is then chosen on calibrated
confidences. The report shows ECE before and after on the held-out half and adds a
`cascade, temperature-scaled` row. Temperature scaling cannot improve AUROC, so this
separates two explanations for over-escalation: if the calibrated cascade is still
expensive, the limit is how well confidence *ranks* answers, not how it is scaled.

### Split robustness (`foveate replay --seeds N`)

Every policy is refit and rescored over N calibration/test splits. The report gives the mean
and standard deviation of accuracy and cost, and the share of splits in which the paired 95%
interval does not put the policy below the reference. A result that holds on one split only
is reported as such.

### Goodput at a latency target (`serve/loadgen.py`)

Median latency and throughput can both look good while most requests miss their deadline.
Following DistServe (Zhong et al., 2024), every load run now records **SLO attainment** (the
share of all requests, failures included, that finish within the target; 30 s by default)
and the report gives each policy's **goodput**: the highest tested arrival rate at which at
least 90% of requests meet the target.

## Evaluation protocol

1. **Held-out everything.** Thresholds, rules, temperatures, router weights, λ and the
   reference config are all chosen on the calibration half. Numbers come from the other half.
2. **Paired comparisons.** Policies answer the same questions, so differences use a paired
   bootstrap over questions, with 95% intervals.
3. **Honest baselines.** Fixed cheapest, static intermediate and fixed strongest are always
   reported, and the oracle is labelled non-deployable.
4. **Costs are measured.** Warm batch-1 latency on vLLM; a policy pays for every run it
   makes, including the router's probe. If timings from different backends would be mixed,
   the replay falls back to model FLOPs for every config.
5. **Same trace per policy.** Load tests replay identical seeded Poisson arrivals.
6. **Negative results stay in.** The cascade's offline loss is in the README next to its
   win under load.

## Experiments still to run

| | Experiment | Needs | Status |
|---|---|---|---|
| E7 | Router and calibrated cascade on MVBench, 20 splits: `foveate replay --dataset mvbench --pilot --seeds 20` | CPU and the cached grid from the October run | implemented, not run |
| E8 | Goodput sweep at 0.1 to 1 req/s for cascade, router and best static | 2x T4 | metric implemented; router not yet in the gateway |
| E9 | Cost-aware queueing in the gateway: serve the request with the smallest predicted cost first, after Fu et al. (2024b), who rank LLM requests by predicted length | 2x T4 | planned |
| E10 | Better certainty signals for the probe: answer margin and entropy (already stored per row), agreement across two frame samplings | grid rerun for the agreement signal | planned |
| E11 | All of MVBench, and a second model family | GPU hours | planned |
| E12 | A second benchmark with long videos, where frame count matters more | GPU hours | planned |

E7 is the cheapest next step and decides the direction: if the router closes a real part of
the oracle gap, E8 and E9 follow; if it does not, E10 is the priority because the probe's
signal is the limit.

## References

- Bai et al. *Qwen2.5-VL Technical Report.* arXiv:2502.13923, 2025.
- Chen, Zaharia and Zou. *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.* arXiv:2305.05176, 2023.
- Chernozhukov et al. *Double/Debiased Machine Learning for Treatment and Structural Parameters.* The Econometrics Journal, 2018.
- Fu et al. *Efficiently Serving LLM Reasoning Programs with Certaindex.* arXiv:2412.20993, 2024. (UC San Diego)
- Fu et al. *Efficient LLM Scheduling by Learning to Rank.* NeurIPS 2024. (UC San Diego)
- Guo, Pleiss, Sun and Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017.
- Kang et al. *NoScope: Optimizing Neural Network Queries over Video at Scale.* VLDB 2017.
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
- Li et al. *MVBench: A Comprehensive Multi-modal Video Understanding Benchmark.* CVPR 2024.
- Ong et al. *RouteLLM: Learning to Route LLMs with Preference Data.* arXiv:2406.18665, 2024.
- Zhong et al. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.* OSDI 2024. (UC San Diego)
