"""Stage 3: compute-allocation policies, evaluated by replaying the grid cache.

Every policy maps a question to the sequence of configs it runs. Accuracy is
that of the last config run; cost is the sum over every config run (a cascade
pays for its failed cheap attempts too).

Policies:
  static      one config for everything (the baselines)
  rules       question keywords -> config, mapping learned on a calibration split
  cascade     start cheap, escalate along a ladder while confidence < tau
  rules+casc  rules pick the starting rung, then the cascade escalates
  oracle      cheapest config that answers correctly (an unreachable upper bound
              that shows how much headroom adaptive allocation has)

Thresholds and rule mappings are fit on a calibration half and reported on a
held-out half, so the headline number is not tuned on its own test data.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import stable_hash
from .stats import auroc, bootstrap_ci, ece, paired_bootstrap_diff
from .tokens import MODEL_SHAPES, prefill_flops, vit_patches
from .utils import read_jsonl

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loading the grid cache
# ---------------------------------------------------------------------------


def load_rows(run_dir: Path, corruption: str | None = None) -> pd.DataFrame:
    rows = []
    for f in sorted((run_dir / "grid").glob("*.jsonl")):
        rows.extend(read_jsonl(f))
    if not rows:
        raise FileNotFoundError(f"no grid rows under {run_dir}/grid; run `foveate grid` first")
    df = pd.DataFrame(rows)
    if "corruption" not in df:
        df["corruption"] = None
    df["corruption"] = df["corruption"].fillna("clean")
    if corruption is not None:
        df = df[df["corruption"] == corruption]
    if "correct" not in df or df["correct"].isna().any():
        missing = df["correct"].isna().sum() if "correct" in df else len(df)
        if missing:
            log.warning("%d rows have no correctness label (LingoQA needs `foveate judge`)", missing)
            df = df[df["correct"].notna()] if "correct" in df else df.iloc[0:0]
    return df.drop_duplicates(["qid", "config", "corruption"], keep="last")


def load_costs(run_dir: Path) -> pd.DataFrame:
    """Per (config, corruption): amortised ms/question and batch-1 latency, merged over shards."""
    acc: dict[tuple, dict] = defaultdict(lambda: {"gen": 0.0, "n": 0, "lat": []})
    for f in sorted((run_dir / "cost").glob("*.json")):
        c = json.loads(f.read_text())
        k = (c["config"], c.get("corruption") or "clean")
        acc[k]["gen"] += c.get("gen_wall_s") or 0.0
        acc[k]["n"] += c.get("n_run") or 0
        acc[k]["lat"].extend(c.get("latency_b1_ms") or [])
    recs = []
    for (cfg, corr), a in acc.items():
        recs.append({
            "config": cfg, "corruption": corr,
            "amortized_ms": a["gen"] / a["n"] * 1e3 if a["n"] else np.nan,
            "latency_b1_ms": float(np.median(a["lat"])) if a["lat"] else np.nan,
            "latency_b1_p95_ms": float(np.percentile(a["lat"], 95)) if a["lat"] else np.nan,
        })
    return pd.DataFrame(recs)


def flops_cost(df: pd.DataFrame, model_ids: dict[str, str]) -> pd.Series:
    """Model-derived prefill TFLOPs per question (used when timings are missing)."""
    out = {}
    for cfg, g in df.groupby("config"):
        model = cfg.split(":")[0]
        shape = MODEL_SHAPES.get(model_ids.get(model, model))
        if shape is None:
            continue
        h, w = g["frame_hw"].iloc[0]
        toks = g["prompt_tokens"].mean()
        out[cfg] = prefill_flops(shape, toks, vit_patches(h, w, int(g["frames"].iloc[0])))["total"] / 1e12
    return pd.Series(out)


@dataclass
class Table:
    qids: np.ndarray
    tasks: np.ndarray
    questions: np.ndarray
    configs: list[str]
    correct: np.ndarray  # [Q, C] float (0/1)
    conf: np.ndarray  # [Q, C]
    cost: np.ndarray  # [C]
    cost_name: str

    def col(self, cfg: str) -> int:
        return self.configs.index(cfg)

    def subset(self, mask: np.ndarray) -> "Table":
        return Table(self.qids[mask], self.tasks[mask], self.questions[mask], self.configs,
                     self.correct[mask], self.conf[mask], self.cost, self.cost_name)


def build_table(
    df: pd.DataFrame, costs: pd.DataFrame, cost_metric: str = "latency_b1_ms",
    corruption: str = "clean", model_ids: dict[str, str] | None = None,
    conf_field: str = "conf",
) -> Table:
    d = df[df["corruption"] == corruption]
    configs = sorted(d["config"].unique(), key=_cfg_key)
    piv_c = d.pivot_table(index="qid", columns="config", values="correct", aggfunc="last")
    piv_p = d.pivot_table(index="qid", columns="config", values=conf_field, aggfunc="last")
    piv_c = piv_c[configs].dropna()
    piv_p = piv_p.loc[piv_c.index, configs].fillna(0.0)
    dropped = d["qid"].nunique() - len(piv_c)
    if dropped:
        log.info("%d questions are missing from at least one config and were dropped", dropped)
    meta = d.drop_duplicates("qid").set_index("qid")
    # Cost per config: measured if available (clean costs reused for shifted
    # runs, since corruption does not change the compute), else model FLOPs.
    cmap = {}
    c = costs[costs["corruption"] == "clean"].set_index("config") if len(costs) else None
    fl = flops_cost(d, model_ids or {})
    name = cost_metric
    for cfg in configs:
        v = np.nan
        if c is not None and cfg in c.index and cost_metric in c:
            v = c.loc[cfg, cost_metric]
        if np.isnan(v):
            v = fl.get(cfg, np.nan)
            name = f"{cost_metric} (TFLOP fallback for some configs)"
        cmap[cfg] = v
    q = piv_c.index.to_numpy()
    return Table(
        qids=q, tasks=meta.loc[q, "task"].to_numpy(),
        questions=meta.loc[q, "question"].to_numpy() if "question" in meta else np.array([""] * len(q)),
        configs=configs, correct=piv_c.to_numpy(dtype=float), conf=piv_p.to_numpy(dtype=float),
        cost=np.array([cmap[c] for c in configs], dtype=float), cost_name=name,
    )


def _cfg_key(name: str):
    model, fr = name.split(":")
    f, r = fr.split("x")
    return (model, int(f), int(r))


def calib_split(t: Table, frac: float = 0.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Stratified (by task) deterministic split into calibration / test masks."""
    calib = np.zeros(len(t.qids), dtype=bool)
    for task in np.unique(t.tasks):
        idx = np.where(t.tasks == task)[0]
        idx = sorted(idx, key=lambda i: stable_hash(f"{seed}:{t.qids[i]}"))
        calib[idx[: int(round(len(idx) * frac))]] = True
    return calib, ~calib


# ---------------------------------------------------------------------------
# Policies. Each returns (final_config_index [Q], cost [Q], n_runs [Q]).
# ---------------------------------------------------------------------------


def static(t: Table, cfg: str):
    j = t.col(cfg)
    n = len(t.qids)
    return np.full(n, j), np.full(n, t.cost[j]), np.ones(n, dtype=int)


def oracle(t: Table):
    order = np.argsort(t.cost)
    final = np.full(len(t.qids), order[0])
    for i in range(len(t.qids)):
        for j in order:
            if t.correct[i, j]:
                final[i] = j
                break
    return final, t.cost[final], np.ones(len(t.qids), dtype=int)


def cascade(t: Table, ladder: list[str], tau: float, start: np.ndarray | None = None):
    """Run ladder rungs in order; stop at the first rung with confidence >= tau."""
    cols = [t.col(c) for c in ladder]
    n = len(t.qids)
    start = np.zeros(n, dtype=int) if start is None else start
    final = np.empty(n, dtype=int)
    cost = np.zeros(n)
    runs = np.zeros(n, dtype=int)
    for i in range(n):
        for k in range(start[i], len(cols)):
            j = cols[k]
            cost[i] += t.cost[j]
            runs[i] += 1
            final[i] = j
            if t.conf[i, j] >= tau:
                break
    return final, cost, runs


CATEGORIES = {
    "count": r"\bhow many\b|\bnumber of\b|\bcount\b|\btimes\b",
    "temporal": r"\bbefore\b|\bafter\b|\bthen\b|\bfirst\b|\blast\b|\bsequence\b|\border\b|"
                r"\bhappen|\bnext\b|\bbegin|\bend\b|\bduring\b|\bwhile\b|\bchange|\btransition",
    "motion": r"\bmov(e|ing|ement)\b|\bdirection\b|\btoward|\brotat|\bturn|\bspeed\b|"
              r"\bcollid|\bcollision\b|\bleft\b|\bright\b|\bbrak|\blane\b",
    "detail": r"\bcolou?r\b|\btext\b|\bread\b|\bsign\b|\bshape\b|\bmaterial\b|\bwearing\b|"
              r"\bholding\b|\blight\b|\bletter\b|\bsmall\b",
}


def categorize(question: str) -> str:
    q = question.lower()
    for name, pat in CATEGORIES.items():
        if re.search(pat, q):
            return name
    return "other"


def fit_rules(t: Table, candidates: list[str], tol: float = 0.01) -> dict[str, str]:
    """Per question category: cheapest candidate within `tol` of that category's best accuracy."""
    cats = np.array([categorize(q) for q in t.questions])
    mapping = {}
    for cat in sorted(set(cats) | {"other"}):
        m = cats == cat
        if not m.any():
            m = np.ones(len(cats), dtype=bool)
        accs = {c: t.correct[m, t.col(c)].mean() for c in candidates}
        best = max(accs.values())
        ok = [c for c in candidates if accs[c] >= best - tol]
        mapping[cat] = min(ok, key=lambda c: t.cost[t.col(c)])
    return mapping


def rules(t: Table, mapping: dict[str, str]):
    cats = [categorize(q) for q in t.questions]
    final = np.array([t.col(mapping.get(c, mapping["other"])) for c in cats])
    return final, t.cost[final], np.ones(len(final), dtype=int)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(t: Table, result, name: str, ref: str | None = None) -> dict:
    final, cost, runs = result
    correct = t.correct[np.arange(len(final)), final]
    acc, lo, hi = bootstrap_ci(correct)
    out = {
        "policy": name, "n": int(len(final)), "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
        "mean_cost": float(np.mean(cost)), "escalation_rate": float(np.mean(runs > 1)),
        "mean_runs": float(np.mean(runs)),
    }
    if ref is not None:
        rc = t.correct[:, t.col(ref)]
        d, dlo, dhi = paired_bootstrap_diff(correct, rc)
        out.update({"vs_ref": ref, "acc_diff": d, "acc_diff_lo": dlo, "acc_diff_hi": dhi,
                    "cost_saving": 1 - float(np.mean(cost)) / t.cost[t.col(ref)]})
    return out


def pareto(points: list[dict]) -> list[dict]:
    pts = sorted(points, key=lambda p: (p["mean_cost"], -p["accuracy"]))
    front, best = [], -1.0
    for p in pts:
        if p["accuracy"] > best:
            front.append(p)
            best = p["accuracy"]
    return front


def confidence_report(t: Table) -> list[dict]:
    """How well does each config's confidence predict its own correctness?"""
    out = []
    for j, c in enumerate(t.configs):
        out.append({"config": c, "accuracy": float(t.correct[:, j].mean()),
                    "auroc": auroc(t.conf[:, j], t.correct[:, j]),
                    "ece": ece(t.conf[:, j], t.correct[:, j]), "cost": float(t.cost[j])})
    return out


def replay(t: Table, ladder: list[str], ref: str | None = None, tol: float = 0.0, seed: int = 0) -> dict:
    """Fit on calibration half, report on test half, and draw full curves."""
    ladder = [c for c in ladder if c in t.configs]
    if not ladder:
        raise ValueError("none of the ladder configs are in the grid")
    ref = ref or max(t.configs, key=lambda c: t.correct[:, t.col(c)].mean())
    calib_m, test_m = calib_split(t, seed=seed)
    cal, test = t.subset(calib_m), t.subset(test_m)

    taus = np.unique(np.quantile(t.conf[:, [t.col(c) for c in ladder]], np.linspace(0, 1, 41)))
    taus = np.concatenate([[0.0], taus, [1.01]])

    # Curves on all data (for the figure).
    curve = []
    for tau in taus:
        r = evaluate(t, cascade(t, ladder, tau), "cascade")
        r["tau"] = float(tau)
        curve.append(r)

    # Threshold: cheapest tau whose calibration accuracy reaches the reference's minus tol.
    target = cal.correct[:, cal.col(ref)].mean() - tol
    cands = [(tau, evaluate(cal, cascade(cal, ladder, tau), "c")) for tau in taus]
    ok = [(tau, r) for tau, r in cands if r["accuracy"] >= target]
    tau_star = min(ok, key=lambda x: x[1]["mean_cost"])[0] if ok else taus[-1]

    mapping = fit_rules(cal, sorted(set(ladder) | {ref}), tol=0.01)
    start_map = {cat: (ladder.index(c) if c in ladder else 0) for cat, c in mapping.items()}
    start_test = np.array([start_map.get(categorize(q), 0) for q in test.questions])
    start_cal = np.array([start_map.get(categorize(q), 0) for q in cal.questions])
    cands_rc = [(tau, evaluate(cal, cascade(cal, ladder, tau, start_cal), "c")) for tau in taus]
    ok_rc = [(tau, r) for tau, r in cands_rc if r["accuracy"] >= target]
    tau_rc = min(ok_rc, key=lambda x: x[1]["mean_cost"])[0] if ok_rc else taus[-1]

    policies = [evaluate(test, static(test, c), f"static {c}", ref) for c in t.configs]
    policies.append(evaluate(test, rules(test, mapping), "rules", ref))
    policies.append(evaluate(test, cascade(test, ladder, tau_star), f"cascade (tau={tau_star:.3f})", ref))
    policies.append(evaluate(test, cascade(test, ladder, tau_rc, start_test),
                             f"rules+cascade (tau={tau_rc:.3f})", ref))
    policies.append(evaluate(test, oracle(test), "oracle (upper bound)", ref))
    # Ablations: drop one rung at a time from the cascade.
    ablations = []
    for drop in ladder[:-1]:
        lad = [c for c in ladder if c != drop]
        ablations.append(evaluate(test, cascade(test, lad, tau_star), f"cascade without {drop}", ref))

    per_task = []
    for task in np.unique(t.tasks):
        m = t.tasks == task
        row = {"task": task, "n": int(m.sum())}
        for c in t.configs:
            row[c] = float(t.correct[m, t.col(c)].mean())
        per_task.append(row)

    return {
        "cost_metric": t.cost_name, "reference": ref, "ladder": ladder, "tau": float(tau_star),
        "tau_rules_cascade": float(tau_rc), "rules": mapping, "n_calib": int(calib_m.sum()),
        "n_test": int(test_m.sum()), "policies": policies, "ablations": ablations,
        "frontier": [p["policy"] for p in pareto(policies)], "curve": curve,
        "confidence": confidence_report(t), "per_task": per_task,
        "costs": dict(zip(t.configs, map(float, t.cost))),
    }
