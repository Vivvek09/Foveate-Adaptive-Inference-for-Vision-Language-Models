"""Stage 3b: learned routing, confidence calibration and split robustness.

All three run on the cached grid (CPU only), like the rest of stage 3.

Why they exist. The first MVBench run showed two things about the confidence
cascade: the cheap rung's confidence only moderately separates right from
wrong answers (AUROC about 0.7), so a safe threshold escalates almost every
question; and a ladder pays for every rung it climbs. The oracle, meanwhile,
is far above every deployable policy. These methods ask how much of that gap
a policy can close using only what it may legitimately see:

  temperature scaling   one scalar per config, fit on the calibration half, so
                        a single threshold means the same thing on every rung
                        (the 7B-AWQ model is overconfident, the 3B is not)
  probe-then-route      run the cheapest rung once as a probe, then jump
                        straight to the config with the best predicted
                        accuracy per unit cost. At most two runs per question,
                        instead of one per ladder rung.
  split robustness      repeat the whole fit/score protocol over many
                        calibration/test splits, so a headline number is not
                        the luck of one split

The router never sees labels at decision time. Its inputs are the probe's
confidence and the question's keyword category; labels are used only to fit
it on the calibration half. The trade-off weight lambda is chosen by
cross-fitting inside the calibration half, so it is not tuned on predictions
the router made for its own training questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np

from .data import stable_hash
from .stats import ece

# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def apply_temperature(conf, temp: float):
    return _sigmoid(_logit(conf) / temp)


def fit_temperature(conf, correct) -> float:
    """The T minimising the log loss of sigmoid(logit(conf) / T) against correctness."""
    z, y = _logit(conf), np.asarray(correct, dtype=float)
    if len(y) == 0:
        return 1.0
    best, best_t = np.inf, 1.0
    for t in np.logspace(-1, 1.3, 93):
        p = np.clip(_sigmoid(z / t), 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best - 1e-12:
            best, best_t = nll, float(t)
    return best_t


def fit_temperatures(t, configs: list[str]) -> dict[str, float]:
    return {c: fit_temperature(t.conf[:, t.col(c)], t.correct[:, t.col(c)]) for c in configs}


def with_temperatures(t, temps: dict[str, float]):
    """A copy of the table whose confidences are temperature-scaled per config."""
    conf = t.conf.copy()
    for c, temp in temps.items():
        conf[:, t.col(c)] = apply_temperature(conf[:, t.col(c)], temp)
    return replace(t, conf=conf)


def calibration_report(t, temps: dict[str, float]) -> list[dict]:
    scaled = with_temperatures(t, temps)
    return [{"config": c, "temperature": float(temp),
             "ece_before": ece(t.conf[:, t.col(c)], t.correct[:, t.col(c)]),
             "ece_after": ece(scaled.conf[:, t.col(c)], t.correct[:, t.col(c)])}
            for c, temp in temps.items()]


# ---------------------------------------------------------------------------
# Probe-then-route
# ---------------------------------------------------------------------------


@dataclass
class Router:
    probe: str
    configs: list[str]
    categories: list[str]
    weights: np.ndarray  # [len(configs), n_features]


def _features(t, probe: str, categories: list[str]) -> np.ndarray:
    from .policy import categorize

    conf = t.conf[:, t.col(probe)]
    cats = np.array([categorize(q) for q in t.questions])
    cols = [np.ones(len(conf)), conf, np.clip(_logit(conf), -8, 8) / 8]
    cols += [(cats == c).astype(float) for c in categories]
    return np.stack(cols, axis=1)


def fit_logreg(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """Ridge-penalised logistic regression by Newton steps (intercept unpenalised)."""
    w = np.zeros(x.shape[1])
    pen = np.full(x.shape[1], l2)
    pen[0] = 1e-6
    for _ in range(iters):
        p = _sigmoid(x @ w)
        grad = x.T @ (p - y) + pen * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        w = w - step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fit_router(t, ladder: list[str], l2: float = 1.0) -> Router:
    """One model per ladder config: P(config answers correctly | what the probe showed)."""
    from .policy import CATEGORIES

    cats = list(CATEGORIES)
    x = _features(t, ladder[0], cats)
    w = np.stack([fit_logreg(x, t.correct[:, t.col(c)], l2) for c in ladder])
    return Router(ladder[0], list(ladder), cats, w)


def predict(router: Router, t) -> np.ndarray:
    """[Q, len(configs)] predicted probability that each config answers correctly."""
    return _sigmoid(_features(t, router.probe, router.categories) @ router.weights.T)


def route(t, router: Router, lam: float, probs: np.ndarray | None = None):
    """Pick argmax of predicted accuracy minus lam x extra cost (as a share of the dearest config).

    Staying with the probe's answer costs nothing extra; any other config costs
    its own run on top of the probe. Returns (final, cost, runs) like every policy.
    """
    probs = predict(router, t) if probs is None else probs
    cols = np.array([t.col(c) for c in router.configs])
    extra = t.cost[cols] / t.cost[cols].max()
    extra[0] = 0.0
    pick = np.argmax(probs - lam * extra[None, :], axis=1)
    final = cols[pick]
    moved = pick != 0
    return final, t.cost[cols[0]] + moved * t.cost[final], 1 + moved.astype(int)


LAMBDAS = np.concatenate([[0.0], np.logspace(-3, 1, 41)])


def crossfit_probs(t, ladder: list[str], folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold router predictions: no question is scored by a model that trained on it."""
    n = len(t.qids)
    folds = max(2, min(folds, n))
    fold = np.array([stable_hash(f"router:{seed}:{q}") % folds for q in t.qids])
    probs = np.zeros((n, len(ladder)))
    for k in range(folds):
        held = fold == k
        if not held.any():
            continue
        train = ~held if (~held).any() else held
        probs[held] = predict(fit_router(t.subset(train), ladder), t.subset(held))
    return probs


def select_lambda(cal, ladder: list[str], target: float, folds: int = 5, seed: int = 0) -> float:
    """Cheapest lambda whose cross-fitted calibration accuracy reaches the target."""
    probs = crossfit_probs(cal, ladder, folds, seed)
    shell = fit_router(cal, ladder)
    best = None
    for lam in LAMBDAS:
        final, cost, _ = route(cal, shell, float(lam), probs)
        acc = cal.correct[np.arange(len(final)), final].mean()
        if acc >= target and (best is None or cost.mean() < best[1]):
            best = (float(lam), cost.mean())
    return best[0] if best else 0.0


def route_shares(t, final: np.ndarray) -> dict[str, float]:
    return {t.configs[j]: float((final == j).mean()) for j in np.unique(final)}


# ---------------------------------------------------------------------------
# Split robustness
# ---------------------------------------------------------------------------


def policy_family(name: str) -> str:
    """'cascade (tau=0.981)' -> 'cascade': the same policy under another split's parameters."""
    return re.sub(r"\s*\(.*\)$", "", name)


def split_robustness(t, ladder: list[str], seeds: list[int], ref: str | None = None,
                     tol: float = 0.0) -> dict:
    """Refit and rescore every policy over many calibration/test splits.

    The reference config is fixed across splits (chosen on the first split's
    calibration half unless given), so every split compares against the same thing.
    """
    from .policy import replay

    runs = []
    for s in seeds:
        r = replay(t, ladder, ref=ref, tol=tol, seed=s)
        ref = ref or r["reference"]
        runs.append(r)
    ref_cost = t.cost[t.col(ref)]
    acc: dict[str, list] = {}
    for r in runs:
        for p in r["policies"]:
            acc.setdefault(policy_family(p["policy"]), []).append(p)
    rows = []
    for name, ps in acc.items():
        a = np.array([p["accuracy"] for p in ps])
        c = np.array([p["mean_cost"] for p in ps]) / ref_cost
        d = np.array([p.get("acc_diff", np.nan) for p in ps])
        rows.append({
            "policy": name, "kind": ps[0]["kind"], "n_splits": len(ps),
            "accuracy_mean": float(a.mean()), "accuracy_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "cost_ratio_mean": float(c.mean()), "cost_ratio_sd": float(c.std(ddof=1)) if len(c) > 1 else 0.0,
            "acc_diff_mean": float(np.nanmean(d)),
            # Share of splits in which the paired 95% interval does not put the policy below the reference.
            "not_worse_share": float(np.mean([p.get("acc_diff_hi", 0.0) >= 0 for p in ps])),
        })
    rows.sort(key=lambda r: r["cost_ratio_mean"])
    return {"reference": ref, "seeds": list(seeds), "ladder": [c for c in ladder if c in t.configs],
            "policies": rows}
