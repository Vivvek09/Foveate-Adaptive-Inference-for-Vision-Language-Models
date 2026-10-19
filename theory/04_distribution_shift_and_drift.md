# 04 · Distribution shift and drift (Stage 4: your original idea, made measurable)

> **One sentence to remember:** a drift detector tells you the inputs changed, not that
> the answers got worse; the cascade is what turns "inputs got harder" into "spend more
> compute on this request".

---

## 1. What kind of shift are we simulating?

Machine learning distinguishes a few kinds of shift between training/reference data
and live traffic:

| Shift | What changes | Driving example |
|---|---|---|
| **covariate** | p(x), the inputs | night, fog, rain, motion blur, a new camera |
| **label / prior** | p(y) | more pedestrians in a city than on a highway |
| **concept** | p(y given x) | a traffic rule changes the right answer |

Foveate simulates **covariate shift** with three corruptions in the style of
ImageNet-C (Hendrycks & Dietterich, ICLR 2019), at severities 1 to 5
(`data/corrupt.py`):

- **fog**: the atmospheric scattering model I = J·t + A·(1−t), with transmission t
  falling toward the top of the frame where the scene is farther away;
- **night**: gamma darkening, a blue cast, and sensor noise that grows as light drops;
- **blur**: horizontal motion blur (ego-motion) plus slight defocus.

They are deterministic (seeded per frame), so a shifted run is reproducible. Real
adverse-condition data (night and rain splits in driving datasets) is the natural next
step.

---

## 2. Three things to measure

`foveate shift` reruns the ladder configs on corrupted frames; `foveate replay` then
reports:

1. **Per corruption:** cheap-rung accuracy, top-rung accuracy, cascade accuracy,
   **escalation rate** and cost, clean versus shifted, using the τ fitted on clean data.
   The hoped-for pattern: escalation rises on shifted inputs, accuracy holds better
   than the cheap config, and cost rises only there.
2. **A stream that switches from clean to shifted halfway**, with a drift detector
   watching it (section 3).
3. **A drift-aware cascade**: while the alarm is on, skip the cheapest rung, because
   most of its answers would be escalated anyway. This saves the wasted cheap call
   on every request during the shift.

---

## 3. A detector that costs nothing

Foveate's detector uses four **model-free image statistics** computed while loading
frames: brightness, contrast, sharpness (variance of the Laplacian) and saturation.
For a sliding window of the last W requests it computes a two-sample z statistic
against a clean reference window:

$$
z = \max_{j} \frac{\left|\bar{x}^{\text{window}}_j - \mu^{\text{ref}}_j\right|}{\sigma^{\text{ref}}_j / \sqrt{W}}
$$

and raises an alarm when z exceeds a threshold (default 6, which is conservative
because the four features are tested together). The report gives **detection delay**
(requests after the switch before the alarm) and **false alarms** on the clean half.

Why not embeddings? A vision-embedding detector (mean shift or MMD on ViT features) is
more general, and is the right upgrade. But it needs the model's hidden states, which
vLLM does not return. The image statistics run on the CPU for free and are enough for
weather and lighting shifts. "Failing Loudly" (Rabanser, Günnemann & Lipton, NeurIPS
2019) found that simple two-sample tests on reduced representations are strong
baselines for shift detection, which supports starting simple.

---

## 4. The three traps in "drift-aware serving"

1. **Drift is not failure.** Fog can shift every statistic and the model can still
   answer correctly. Only labels show accuracy, which is why Foveate reports accuracy
   on shifted copies of labelled questions rather than trusting the detector.
2. **Wrong time scale.** A window statistic reacts after W requests. Per-request
   decisions (escalate this one) must come from per-request signals such as confidence.
   The detector's job is the slow loop: change the *default* starting rung.
3. **Confidence degrades under shift too.** Ovadia et al. (NeurIPS 2019, "Can You
   Trust Your Model's Uncertainty?") showed calibration gets worse as shift grows. The
   report's `shifted_cheap_auroc` column checks this for your model. If AUROC drops
   from 0.75 to 0.6 under fog, the cascade's threshold picks the wrong questions
   exactly when you need it most, and the drift-aware start rung becomes more valuable.

---

## 5. Correlations worth reporting

These come straight out of the shift results:

- **Corruption severity vs. accuracy drop** per config: do more frames or more
  pixels make the model more robust? (Blur often hurts low resolution less, since the
  detail was already gone.)
- **Escalation-rate increase vs. accuracy drop** across corruptions: if they move
  together, confidence is tracking difficulty.
- **Drift z vs. cheap-rung confidence** over the stream: does the model-free signal
  lead the model-based one?

> **Trend check.** Physical-AI teams treat this as "long-tail" or "out-of-distribution"
> scenario handling: rare weather, lighting and sensor conditions dominate failure
> analysis. Validation and safety teams increasingly use VLMs to *mine* such scenarios
> from fleet logs, which is exactly where inference cost per clip matters. Keep the
> claim modest: recorded-clip QA under synthetic corruption is a robustness study,
> not evidence of on-road safety.

**Read next:** ImageNet-C (arXiv 1903.12261); Failing Loudly (arXiv 1810.11953);
Ovadia et al. (arXiv 1906.02530); then note 05.

---

## 6. Measured results (pilot run, Kaggle 2x T4, Oct 19 2026)

From `results/REPORT.md`, MVBench pilot, severity 3, τ fitted on clean data.

| Corruption | Cascade acc (clean → shifted) | Escalation | Cheap-rung acc | Cheap-rung AUROC | Cost (ms) |
|---|---|---|---|---|---|
| blur | 0.671 → 0.609 | 0.928 → 0.970 | 0.556 → 0.496 | 0.714 → 0.656 | 2,429 → 2,628 |
| night | 0.671 → 0.602 | 0.928 → 0.989 | 0.556 → 0.478 | 0.714 → 0.658 | 2,429 → 2,734 |
| fog | 0.671 → 0.655 | 0.928 → 0.956 | 0.556 → 0.526 | 0.714 → 0.707 | 2,429 → 2,498 |

Drift stream (clean, then shifted):

| Corruption | Alarm delay | False alarms (clean) | Drift-aware vs plain cascade (shifted half) |
|---|---|---|---|
| night | 17 requests | 0 | cost 2,606 vs 2,749 ms (−5.2%), acc 0.595 vs 0.597 |
| fog | 39 requests | 0 | cost 2,384 vs 2,484 ms (−4.0%), acc 0.653 vs 0.655 |
| blur | never fired | 0 | identical (alarm never on) |

- **Escalation rose under every corruption** (most under night), so confidence tracks
  difficulty, and cost rose only on shifted inputs.
- **Trap 3 confirmed:** the cheap rung's AUROC fell from 0.714 to ~0.66 under blur and
  night, so the threshold picks worse questions to escalate exactly under shift.
- **Escalation cannot undo a corrupted input:** the 7B sees the same frames, so cascade
  accuracy still fell 6 to 7 points under blur and night.
- **The free detector missed blur.** A likely reason (not tested): sharpness varies a lot
  between clean MVBench clips, so the reference spread is wide. An embedding-based
  detector is the upgrade.
