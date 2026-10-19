"""Stage 4: distribution shift. Does the cascade spend more where inputs get harder,
and can a cheap drift monitor tell the cascade to start higher?

Three measurements, all replayed from the grid cache:
  1. Per corruption: accuracy, escalation rate and cost of the cascade (with the
     threshold fit on clean data) next to the cheapest and most expensive static
     configs. The cascade should escalate more on shifted inputs.
  2. A simulated stream that switches from clean to shifted halfway. A window
     detector on model-free image statistics (brightness, contrast, sharpness,
     saturation) raises an alarm; escalation rate and confidence are tracked too.
  3. Drift-aware cascade: while the alarm is on, skip the cheapest rung (its
     answers are mostly escalated anyway). Compare cost and accuracy on the
     shifted half with the plain cascade.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .policy import Table, build_table, cascade, evaluate
from .stats import auroc

log = logging.getLogger(__name__)

STAT_COLS = ["img_brightness", "img_contrast", "img_sharpness", "img_saturation"]


def window_z(ref: np.ndarray, win: np.ndarray) -> float:
    """Largest standardized mean shift across features (a two-sample z statistic)."""
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    z = (win.mean(0) - mu) / (sd / np.sqrt(len(win)))
    return float(np.max(np.abs(z)))


def stream_experiment(
    df: pd.DataFrame, clean: Table, shifted: Table, corruption: str, ladder: list[str],
    tau: float, window: int = 50, z_alarm: float = 6.0, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    common = np.intersect1d(clean.qids, shifted.qids)
    if len(common) < 2 * window:
        window = max(5, len(common) // 4)
    perm = rng.permutation(common)
    half = len(perm) // 2
    stream_q = np.concatenate([perm[:half], perm[half:]])
    is_shift = np.r_[np.zeros(half, bool), np.ones(len(perm) - half, bool)]

    cheap = ladder[0]
    feats = {}
    for corr in ("clean", corruption):
        d = df[(df["corruption"] == corr) & (df["config"] == cheap)].set_index("qid")
        feats[corr] = d[STAT_COLS]
    X = np.stack([
        feats[corruption if sh else "clean"].loc[q].to_numpy(float) for q, sh in zip(stream_q, is_shift)
    ])
    ref = X[:window]

    # Per-question cascade outcome on the stream, plain and drift-aware.
    def outcome(tab: Table, q: str, start: int):
        i = int(np.where(tab.qids == q)[0][0])
        sub = tab.subset(np.arange(len(tab.qids)) == i)
        f, c, r = cascade(sub, ladder, tau, start=np.array([start]))
        return float(sub.correct[0, f[0]]), float(c[0]), int(r[0]), float(sub.conf[0, sub.col(cheap)])

    z_trace, alarm, plain, aware = [], [], [], []
    on = False
    for k, (q, sh) in enumerate(zip(stream_q, is_shift)):
        if k >= window:
            z = window_z(ref, X[k - window:k])
            on = z > z_alarm
        else:
            z = 0.0
        z_trace.append(z)
        alarm.append(on)
        tab = shifted if sh else clean
        plain.append(outcome(tab, q, 0))
        aware.append(outcome(tab, q, 1 if on and len(ladder) > 1 else 0))
    plain_a, aware_a = np.array(plain), np.array(aware)
    first_alarm = next((k for k, a in enumerate(alarm) if a and is_shift[k]), None)
    false_alarms = int(np.sum(np.array(alarm) & ~is_shift))

    def seg(arr, m):
        return {"accuracy": float(arr[m, 0].mean()), "mean_cost": float(arr[m, 1].mean()),
                "escalation_rate": float((arr[m, 2] > 1).mean())}

    return {
        "corruption": corruption, "window": window, "z_alarm": z_alarm,
        "detection_delay": (first_alarm - half) if first_alarm is not None else None,
        "false_alarms_on_clean": false_alarms,
        "plain": {"clean": seg(plain_a, ~is_shift), "shifted": seg(plain_a, is_shift)},
        "drift_aware": {"clean": seg(aware_a, ~is_shift), "shifted": seg(aware_a, is_shift)},
        "trace": {
            "z": z_trace, "alarm": alarm, "is_shift": is_shift.tolist(),
            "escalated": (plain_a[:, 2] > 1).astype(int).tolist(),
            "cheap_conf": plain_a[:, 3].tolist(),
        },
    }


def shift_report(df: pd.DataFrame, costs: pd.DataFrame, ladder: list[str], tau: float,
                 model_ids: dict[str, str]) -> dict:
    corrs = [c for c in df["corruption"].unique() if c != "clean"]
    if not corrs:
        raise ValueError("no shifted rows; run `foveate grid --corruption fog:3` (or `foveate shift`)")
    clean = build_table(df, costs, corruption="clean", model_ids=model_ids)
    out = {"tau": tau, "ladder": ladder, "per_corruption": [], "streams": []}
    for corr in sorted(corrs):
        sh = build_table(df, costs, corruption=corr, model_ids=model_ids)
        lad = [c for c in ladder if c in sh.configs and c in clean.configs]
        sh.cost = np.array([clean.cost[clean.col(c)] if c in clean.configs else np.nan for c in sh.configs])
        cm = np.isin(clean.qids, sh.qids)
        cl = clean.subset(cm)
        rec = {"corruption": corr, "n": int(len(sh.qids))}
        for tag, tab in (("clean", cl), ("shifted", sh)):
            r = evaluate(tab, cascade(tab, lad, tau), "cascade")
            rec[f"{tag}_cascade_acc"] = r["accuracy"]
            rec[f"{tag}_escalation"] = r["escalation_rate"]
            rec[f"{tag}_cost"] = r["mean_cost"]
            rec[f"{tag}_cheap_acc"] = float(tab.correct[:, tab.col(lad[0])].mean())
            rec[f"{tag}_top_acc"] = float(tab.correct[:, tab.col(lad[-1])].mean())
            rec[f"{tag}_cheap_conf"] = float(tab.conf[:, tab.col(lad[0])].mean())
            rec[f"{tag}_cheap_auroc"] = auroc(tab.conf[:, tab.col(lad[0])], tab.correct[:, tab.col(lad[0])])
        out["per_corruption"].append(rec)
        out["streams"].append(stream_experiment(df, cl, sh, corr, lad, tau))
    return out
