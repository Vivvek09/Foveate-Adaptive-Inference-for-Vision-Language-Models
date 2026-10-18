# 03 · Adaptive compute and cascades (Stage 3: policies)

> **One sentence to remember:** most questions are easy, so answer everything cheaply
> first and pay for more frames, pixels or parameters only when the cheap answer looks
> unsure.

---

## 1. The optimisation problem

For each question *x*, a policy π chooses which configs to run. We want

$$
\min_{\pi}\ \mathbb{E}_x\big[\text{cost}(\pi, x)\big]
\quad \text{s.t.} \quad
\text{accuracy}(\pi) \ge \text{accuracy}(\text{reference}) - \delta
$$

where the reference is the best fixed config and δ is the accuracy you are willing to
give up (`--tol`, default 0). Foveate compares five kinds of policy, all by replaying
the grid cache:

| Policy | Decides with | Cost |
|---|---|---|
| static | nothing: one config for all | cost of that config |
| rules | question keywords → config | one config per question |
| cascade | confidence after each rung | sum of every rung it ran |
| rules + cascade | keywords pick the first rung, confidence escalates | sum of rungs run |
| oracle | the answer key | cheapest correct config (unreachable bound) |

The **oracle** is not a policy you can deploy. It tells you the *headroom*: if the
oracle saves 70% and your cascade saves 30%, the gap is a better confidence signal
away.

---

## 2. The cascade's cost model

With a cheap rung of cost c₁, an expensive rung of cost c₂ and escalation rate p
(the share of questions whose cheap confidence is below τ):

$$
\mathbb{E}[\text{cost}] = c_1 + p\,c_2
$$

It beats always running the expensive rung when

$$
c_1 + p\,c_2 < c_2 \iff p < 1 - \frac{c_1}{c_2}
$$

Worked example from the default ladder: 4 frames at 224 is 128 visual tokens, and 16
frames at 448 is 2,048. If latency scales with tokens, c₁/c₂ ≈ 0.06, so the cascade
still saves money even if it escalates **94%** of questions. Cheap rungs this cheap
are almost free insurance. The real constraint is accuracy: escalating on the wrong
questions, or *not* escalating on confidently-wrong ones, is what loses points.

**Failure mode to watch:** confidently wrong cheap answers. If the cheap rung answers
"A" with 0.9 confidence and is wrong, no threshold rescues it. The AUROC column of the
cheap config bounds how good any cascade can be.

---

## 3. Picking τ without fooling yourself

If you sweep τ on the test questions and report the best point, you have tuned on the
test set and the number is optimistic. Foveate splits the questions into a
**calibration half** and a **held-out half** (stratified by task, fixed seed):

1. Sweep τ over quantiles of the confidence on the calibration half.
2. Pick the cheapest τ whose calibration accuracy reaches the reference's accuracy
   minus δ.
3. Report accuracy and cost **on the held-out half**, with a paired-bootstrap interval
   on the accuracy difference versus the reference.

The full curve over all τ (drawn on all questions) appears in the frontier figure for
intuition. The held-out points are the claims.

> **Going further: guarantees.** Conformal risk control (Angelopoulos et al., 2022)
> turns step 2 into a statement like "with probability 1−α, held-out accuracy loss is
> at most δ" under exchangeability. It is a natural upgrade once the basic cascade
> works.

---

## 4. Rules: using the question text

The question already says something about what evidence it needs:

| Category | Keyword cues | Hypothesis |
|---|---|---|
| temporal | before, after, then, sequence, order, happen | needs more **frames** |
| motion | moving, direction, toward, turn, collide, lane | needs more **frames** |
| count | how many, number of, times | frames and pixels |
| detail | colour, text, sign, shape, light, small | needs more **pixels** |
| other | anything else | default |

Foveate does not hard-code the mapping. `fit_rules` learns, per category on the
calibration half, the **cheapest config within 1 point of that category's best**. The
learned mapping is printed in the report. If "temporal" maps to 16 frames at 224 and
"detail" maps to 8 frames at 672, the hypothesis holds; if not, that is a finding too.

Rules decide before seeing any answer (no wasted cheap call). Cascades decide after
seeing one (better information, wasted cheap call). **Rules + cascade** uses the
rule to pick the starting rung, then escalates.

---

## 5. Ablations

The report's ablation table removes one rung at a time from the cascade (keeping τ
fixed). If dropping the middle rung costs no accuracy, the ladder is too long; if
dropping the 7B rung costs a lot, model size is doing work that frames and pixels
cannot.

---

## 6. Where this sits in current research and industry

Adaptive compute is one of the main ideas in inference right now, at every level:

- **Model cascades and routers.** FrugalGPT (Chen, Zaharia & Zou, 2023) cascades API
  models by a learned answer scorer; RouteLLM (LMSYS, 2024) trains routers between a
  strong and a weak LLM from preference data. Foveate applies the idea to the *visual
  input* (frames, pixels) as well as model size.
- **Early exit inside a model.** CALM (Schuster et al., NeurIPS 2022) stops computing
  layers once a token's prediction is confident. Same principle, finer grain.
- **Test-time compute scaling.** Reasoning models (OpenAI o1, DeepSeek-R1) spend more
  tokens on harder prompts, and Snell et al. (2024) showed compute-optimal allocation
  by difficulty beats uniform allocation. A cascade is the cheapest version of
  "difficulty-aware compute".
- **Video-specific.** Query-aware frame selection and token pruning for long video are
  an active line. The novelty warning from the project plan stands: Foveate's claim is
  not "adaptive frames is new", it is a **measured** cost/accuracy trade-off on a real
  serving stack, with held-out evaluation and an honest oracle gap.

---

## 7. What a good Stage 3 result looks like

A sentence you can defend, for example:

> "On the MVBench held-out half, the rules+cascade policy matched the best fixed
> config's accuracy (difference −0.4 points, 95% CI [−1.5, +0.7]) at 38% lower mean
> batch-1 latency, escalating 41% of questions."

The numbers in that sentence are placeholders. Every number in your README must come
from `results/REPORT.md`. If the cascade does *not* beat the best static config, say
so and use the confidence AUROC and oracle gap to explain why. That is also a result.

**Read next:** FrugalGPT (arXiv 2305.05176); RouteLLM (arXiv 2406.18665); then note 04.
