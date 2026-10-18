"""Statistics used everywhere: bootstrap intervals, paired tests, calibration."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0, stat=np.mean):
    """Percentile bootstrap interval for a statistic of one sample."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def paired_bootstrap_diff(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean(a - b) with a bootstrap interval, resampling *questions* (pairs).

    Pairing matters: two configs answering the same questions are strongly
    correlated, so the interval on their difference is much tighter than the
    two separate intervals suggest.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)


def auroc(scores, labels) -> float:
    """Probability a random correct answer has higher confidence than a random wrong one."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=float)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(conf, correct, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    c = np.asarray(conf, dtype=float)
    y = np.asarray(correct, dtype=float)
    bins = np.minimum((c * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - c[m].mean())
    return float(total)


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def linear_fit(x, y) -> dict:
    """Least squares y = a + b x, with R^2 (used for latency vs visual tokens)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"intercept": float(a), "slope": float(b), "r2": float(1 - ss_res / ss_tot) if ss_tot else float("nan")}
