# 02 · Benchmarks, confidence and honest statistics (Stage 2: the quality grid)

> **One sentence to remember:** run every question through every config once, cache
> the answer, the confidence and the cost, and you can evaluate any policy later in
> seconds instead of GPU hours.

---

## 1. Why MVBench is the main benchmark

The ask was a dataset that is widely used and fast to run. MVBench (Li et al., CVPR
2024) is both:

- **Widely used.** It sits in the video table of almost every recent open VLM report.
  The Qwen2.5-VL report gives **67.0** for the 3B model and **69.6** for the 7B. That
  gives you a published number to sanity-check your harness against.
- **Fast.** 20 tasks × 200 questions, short clips, and every answer is one option
  letter, so a question costs one prefill and *one* decode step.
- **Temporal by design.** Its tasks (action sequence, moving direction, object
  shuffle, counterfactual inference...) were built from existing datasets so that a
  single frame should not be enough. That makes the frames knob meaningful.

Two honest caveats:

1. **Your accuracy will be lower than 67.0.** Qwen evaluated with up to 768 frames at
   2 fps. Foveate deliberately uses 4 to 16 frames, because the project is about the
   cost/accuracy trade-off. Being 5 to 10 points under at 16 frames is expected; being
   30 points under means a bug.
2. **One task is missing.** `fine_grained_pose` uses NTU RGB+D videos, which are not
   redistributed on the Hub. Foveate runs the other 19. `--tasks lite` drops 4 more
   (the 11 GB Perception Test zip) when disk or time is tight.

**LingoQA** (Wayve, ECCV 2024) is the driving demo: 500 questions on 100 driving
clips, free-form answers scored by Lingo-Judge, a small classifier the authors trained
because BLEU-style metrics correlate badly with human judgement on this data.

> **Trend check.** Benchmarks leak shortcuts. "Revealing Single Frame Bias for
> Video-and-Language Learning" (Lei et al., 2022) showed many video QA sets are mostly
> solvable from one frame, and MMStar (2024) showed many image benchmarks are partly
> solvable with no image at all. Foveate's per-task heatmap tests this directly: a task
> whose accuracy does not move from 4 to 16 frames is not really temporal for this model.

---

## 2. Scoring multiple choice from logits

Instead of generating text and parsing it, Foveate reads the next-token
log-probabilities of the option letters and renormalises over the valid ones:

$$
p(\text{A}) = \frac{e^{\ell_A}}{\sum_{k \in \text{options}} e^{\ell_k}}
$$

The prediction is the most probable letter. This removes parsing errors, costs one
decode step, and gives three **confidence signals** for free:

| Signal | Definition | Intuition |
|---|---|---|
| `conf` | max option probability | how sure the model is |
| `margin` | top-1 minus top-2 probability | how close the runner-up is |
| `entropy` | −Σ p log p over options | how spread out the belief is |

For free-form answers (LingoQA) the signal is the **geometric-mean token probability**
of the generated answer, exp(mean log-prob). It is weaker: a fluent wrong sentence can
be confident.

---

## 3. Is the confidence any good? AUROC and ECE

The cascade in note 03 only works if confidence separates right answers from wrong
ones. Two numbers in `REPORT.md` measure that per config:

- **AUROC**: the probability that a randomly chosen correct answer has higher
  confidence than a randomly chosen wrong one. 0.5 is useless; 0.8 is good.
  This is what the cascade needs, since it only *ranks* questions by confidence.
- **ECE** (expected calibration error): bin answers by confidence and compare each
  bin's average confidence with its accuracy. Low ECE means "70% sure" really is right
  70% of the time. The cascade does not need calibration, only ranking, but
  calibration matters if you show confidences to users.

Modern instruction-tuned models are usually **overconfident**: RLHF and instruction
tuning sharpen the output distribution. Expect ECE well above zero and AUROC between
0.65 and 0.8.

---

## 4. The pilot subset: 30% that you can trust

`--subset pilot` picks a **fixed-seed, task-stratified 30%** (≈1,140 MVBench questions,
150 LingoQA questions). Why these choices:

- **Stratified** so every task keeps ~60 questions and per-task accuracy is still
  estimable.
- **Fixed seed** so pilot runs on different days hit the same questions and can be
  compared directly.
- **How precise is a pilot?** The standard error of an accuracy p on n questions is
  √(p(1−p)/n). At p = 0.6 and n = 1,140 that is ±1.5 points (±2.9 for a 95%
  interval). Good enough to rank configs that differ by 3+ points; run `--full` for
  the headline numbers (±1.6 points at 95%).

---

## 5. Bootstrap intervals and *paired* comparisons

Every accuracy in the report carries a **bootstrap 95% interval**: resample questions
with replacement 2,000 times, recompute, and take the 2.5th and 97.5th percentiles.

The important subtlety is **pairing**. Two configs answer the *same* questions, and a
hard question is hard for both. So the interval on their *difference* is computed by
resampling questions once and differencing within each resample
(`paired_bootstrap_diff`). That interval is often 2 to 3× tighter than you would guess
by comparing two separate intervals, which is exactly what lets a pilot detect a
2-point difference.

---

## 6. Cost: two kinds of milliseconds

The grid records cost per config in two ways (`results/<run>/cost/*.json`):

| Metric | What it is | Use it for |
|---|---|---|
| `latency_b1_ms` | median batch-1 end-to-end time from a small probe | what one user waits |
| `amortized_ms` | batched GPU wall time ÷ questions | what the GPU pays (throughput cost) |

They can disagree: batching helps cheap configs more than expensive ones, because
cheap configs leave the GPU underused at batch 1. `foveate replay --cost amortized_ms`
switches the policy analysis to throughput cost.

---

## 7. Why caching the grid is the key engineering move

The grid is a matrix of `questions × configs`, each cell holding (correct, confidence,
tokens). Any policy (a fixed config, a rule, a cascade, a drift response) is a
function that walks this matrix. So:

- trying a new policy costs **seconds**,
- every policy is evaluated on **exactly the same answers** (a paired comparison), and
- live serving (note 05) only has to confirm that replayed numbers survive real load.

This "simulate from a log" pattern is how production teams evaluate routing and
caching changes before shipping them.

**Read next:** the MVBench paper (arXiv 2311.17005), section 3 on task construction;
Guo et al. 2017, "On Calibration of Modern Neural Networks"; then note 03.

---

## 8. Measured results (pilot run, Kaggle 2x T4, Oct 19 2026)

From `results/REPORT.md`: MVBench pilot, 1,135 questions over 19 tasks (14 skipped for
missing media), split into 568 calibration and 567 held-out questions.

- **Harness sanity check passes.** 3B at 16 × 448 scored 0.658 and 7B-AWQ at 16 × 448
  scored 0.683 on the held-out half, close to the published 67.0 and 69.6 despite far
  fewer frames. Section 1 predicted "5 to 10 points under"; the gap was smaller.
- **Confidence quality, held-out half:**

| Config | Accuracy | AUROC | ECE |
|---|---|---|---|
| 3b:4x224 | 0.575 | 0.707 | 0.074 |
| 3b:8x448 | 0.654 | 0.785 | 0.049 |
| 3b:16x448 | 0.658 | 0.798 | 0.063 |
| 7b:8x448 | 0.654 | 0.715 | 0.225 |
| 7b:16x448 | 0.683 | 0.725 | 0.197 |

- **Section 3's prediction held for the 7B, not the 3B.** The 3B is well calibrated
  (ECE 0.05 to 0.07) and ranks well (AUROC up to 0.80). The 7B-AWQ is overconfident
  (ECE ~0.20) and ranks worse (0.72). Whether AWQ quantisation contributes was not tested.
- **Precision.** With n = 567 held-out questions, 95% intervals are about ±4 points;
  paired intervals on differences are about ±3 points (e.g. rules vs 7B: −1.1 [−3.9, +1.9]).
- **Two costs disagree, as section 6 warned.** 3b:16x448 costs 1,042 ms batch-1 but
  1,979 ms amortised per question in the batched grid; 7b:16x448 costs 1,625 vs 2,969 ms.
- **LingoQA pilot** (150 questions, 75 held out) is too small: intervals are about ±11
  points and no policy differs significantly from another.
