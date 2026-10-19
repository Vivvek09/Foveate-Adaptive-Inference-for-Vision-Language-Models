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

Thresholds, rule mappings and the reference config are all chosen on a
calibration half and reported on a held-out half, so the headline number is
not tuned on its own test data. Routing only ever reads confidences and the
question text, never labels; labels are used to fit tau/rules on the
calibration half and to score.

Named baselines (all static, chosen from the ladder so they are deployable):
  fixed cheapest      first ladder rung
  static intermediate middle ladder rung
  fixed strongest     last ladder rung (usually the 7B model)
The oracle is reported as an upper bound and is NOT deployable: it peeks at
correctness to pick the cheapest config that answers right.
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
    acc: dict[tuple, dict] = defaultdict(lambda: {"gen": 0.0, "n": 0, "lat": [], "backend": set(),
                                                  "measured": True})
    for f in sorted((run_dir / "cost").glob("*.json")):
        c = json.loads(f.read_text())
        k = (c["config"], c.get("corruption") or "clean")
        acc[k]["gen"] += c.get("gen_wall_s") or 0.0
        acc[k]["n"] += c.get("n_run") or 0
        acc[k]["lat"].extend(c.get("latency_b1_ms") or [])
        acc[k]["backend"].add(str(c.get("backend")))
        acc[k]["measured"] &= bool(c.get("measured", False))
    recs = []
    for (cfg, corr), a in acc.items():
        recs.append({
            "config": cfg, "corruption": corr, "backend": "+".join(sorted(a["backend"])),
            "measured": a["measured"],
            "amortized_ms": a["gen"] / a["n"] * 1e3 if a["n"] else np.nan,
            "latency_b1_ms": float(np.median(a["lat"])) if a["lat"] else np.nan,
            "latency_b1_p95_ms": float(np.percentile(a["lat"], 95)) if a["lat"] else np.nan,
        })
    return pd.DataFrame(recs)


def flops_cost(df: pd.DataFrame, model_ids: dict[str, str]) -> pd.Series:
    """Model-derived prefill TFLOPs per question (used when timings are missing)."""
    out = {}
    ids = {"3b": "Qwen/Qwen2.5-VL-3B-Instruct", "7b": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"}
    ids.update(model_ids or {})
    for cfg, g in df.groupby("config"):
        model = cfg.split(":")[0]
        shape = MODEL_SHAPES.get(ids.get(model, model))
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
    cost_note: str = ""

    def col(self, cfg: str) -> int:
        return self.configs.index(cfg)

    def subset(self, mask: np.ndarray) -> "Table":
        return Table(self.qids[mask], self.tasks[mask], self.questions[mask], self.configs,
                     self.correct[mask], self.conf[mask], self.cost, self.cost_name, self.cost_note)


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
    # Cost per config: measured batch-1 latency (clean costs reused for shifted
    # runs, since corruption does not change the compute) when every config has
    # one from the same real backend; otherwise model-derived prefill TFLOPs for
    # ALL configs, so the units are never mixed.
    c = costs[costs["corruption"] == "clean"].set_index("config") if len(costs) else None
    fl = flops_cost(d, model_ids or {})
    note = ""
    measured = {}
    if c is not None and cost_metric in c:
        for cfg in configs:
            if cfg in c.index and not np.isnan(c.loc[cfg, cost_metric]):
                measured[cfg] = float(c.loc[cfg, cost_metric])
    backends = set(c.loc[[x for x in configs if x in c.index], "backend"]) \
        if c is not None and "backend" in c else set()
    real = c is None or "measured" not in c or bool(
        c.loc[[x for x in configs if x in c.index], "measured"].all())
    if len(measured) == len(configs) and len(backends) <= 1 and real:
        cmap, name = measured, cost_metric
        if backends:
            name = f"{cost_metric} [{next(iter(backends))}]"
    else:
        cmap, name = {cfg: fl.get(cfg, np.nan) for cfg in configs}, "prefill TFLOPs (model-derived)"
        if len(measured) < len(configs):
            note = (f"measured {cost_metric} missing for "
                    f"{sorted(set(configs) - set(measured))}; using model FLOPs for every config")
        elif len(backends) > 1:
            note = (f"configs were timed on different backends {sorted(backends)}; timings are "
                    "not comparable, using model FLOPs for every config")
        else:
            note = "costs are not real measurements (mocked backend); using model FLOPs"
        log.warning(note)
    q = piv_c.index.to_numpy()
    return Table(
        qids=q, tasks=meta.loc[q, "task"].to_numpy(),
        questions=meta.loc[q, "question"].to_numpy() if "question" in meta else np.array([""] * len(q)),
        configs=configs, correct=piv_c.to_numpy(dtype=float), conf=piv_p.to_numpy(dtype=float),
        cost=np.array([cmap[c] for c in configs], dtype=float), cost_name=name, cost_note=note,
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


def baselines(ladder: list[str]) -> dict[str, str]:
    """Deployable static baselines named by their place on the ladder."""
    return {"fixed cheapest": ladder[0], "static intermediate": ladder[len(ladder) // 2],
            "fixed strongest": ladder[-1]}


def escalation_analysis(t: Table, ladder: list[str], tau: float, start: np.ndarray | None = None) -> dict:
    """What escalation bought, per question, on the given (held-out) table.

    helped:  wrong at the first rung run, right at the final rung
    hurt:    right at the first rung run, wrong at the final rung (escalation
             replaced a correct cheap answer by a wrong expensive one)
    overconfident: stopped early (conf >= tau) and wrong; escalation never got a chance
    """
    cols = [t.col(c) for c in ladder]
    n = len(t.qids)
    start = np.zeros(n, dtype=int) if start is None else start
    final, cost, runs = cascade(t, ladder, tau, start)
    first = np.array([cols[start[i]] for i in range(n)])
    c_first = t.correct[np.arange(n), first].astype(bool)
    c_final = t.correct[np.arange(n), final].astype(bool)
    esc = runs > 1
    stopped_wrong = (~esc) & (~c_final) & (final != cols[-1])
    conf_final = t.conf[np.arange(n), final]
    out = {
        "n": int(n), "escalated": int(esc.sum()), "escalation_rate": float(esc.mean()),
        "helped": int((esc & ~c_first & c_final).sum()),
        "hurt": int((esc & c_first & ~c_final).sum()),
        "escalated_no_change_right": int((esc & c_first & c_final).sum()),
        "escalated_no_change_wrong": int((esc & ~c_first & ~c_final).sum()),
        "overconfident_wrong": int(stopped_wrong.sum()),
        "overconfident_wrong_rate": float(stopped_wrong.mean()),
        "accuracy_if_never_escalated": float(c_first.mean()),
        "accuracy": float(c_final.mean()),
        "extra_cost_from_escalation": float(cost.mean() - t.cost[first].mean()),
        "mean_conf_wrong_accepted": float(conf_final[stopped_wrong].mean()) if stopped_wrong.any()
        else None,
    }
    out["net_gain"] = out["helped"] - out["hurt"]
    # The most confident wrong answers the cascade accepted: what to look at by hand.
    idx = np.where(stopped_wrong)[0]
    idx = idx[np.argsort(-conf_final[idx])][:10]
    out["examples_overconfident"] = [
        {"qid": str(t.qids[i]), "task": str(t.tasks[i]), "config": t.configs[final[i]],
         "conf": float(conf_final[i])} for i in idx]
    hurt_idx = np.where(esc & c_first & ~c_final)[0][:10]
    out["examples_hurt"] = [
        {"qid": str(t.qids[i]), "task": str(t.tasks[i]), "first": t.configs[first[i]],
         "final": t.configs[final[i]]} for i in hurt_idx]
    return out


def verdicts(policies: list[dict], names: dict[str, str]) -> list[str]:
    """Plain-language comparisons of each adaptive policy with each baseline (paired, held-out)."""
    by = {p["policy"]: p for p in policies}
    out = []
    adaptive = [p for p in policies if p["policy"].startswith(("cascade", "rules"))]
    for a in adaptive:
        for bname, cfg in names.items():
            b = by.get(f"static {cfg}")
            if b is None:
                continue
            d = paired_bootstrap_diff(np.asarray(a["_correct"]), np.asarray(b["_correct"]))
            cost_ratio = a["mean_cost"] / b["mean_cost"] if b["mean_cost"] else float("nan")
            sig = "higher" if d[1] > 0 else "lower" if d[2] < 0 else "not significantly different"
            out.append(f"{a['policy']} vs {bname} ({cfg}): accuracy {d[0]:+.3f} "
                       f"[{d[1]:+.3f}, {d[2]:+.3f}] ({sig}), cost x{cost_ratio:.2f}")
    return out


def replay(t: Table, ladder: list[str], ref: str | None = None, tol: float = 0.0, seed: int = 0) -> dict:
    """Fit everything on the calibration half; report on the held-out half."""
    ladder = [c for c in ladder if c in t.configs]
    if not ladder:
        raise ValueError("none of the ladder configs are in the grid")
    calib_m, test_m = calib_split(t, seed=seed)
    cal, test = t.subset(calib_m), t.subset(test_m)
    # Reference = most accurate config on the calibration half (not on the test half).
    ref = ref or max(t.configs, key=lambda c: cal.correct[:, cal.col(c)].mean())

    taus = np.unique(np.quantile(cal.conf[:, [cal.col(c) for c in ladder]], np.linspace(0, 1, 41)))
    taus = np.concatenate([[0.0], taus, [1.01]])

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

    # The trade-off curve over every threshold, on the held-out half (descriptive:
    # it shows the whole frontier; the operating point above was picked on calib).
    curve = []
    for tau in taus:
        r = evaluate(test, cascade(test, ladder, tau), "cascade")
        r["tau"] = float(tau)
        curve.append(r)

    names = baselines(ladder)
    label = {cfg: [] for cfg in t.configs}
    for bname, cfg in names.items():
        label[cfg].append(bname)
    if ref not in names.values():
        label[ref].append("reference")

    def ev(result, name, kind):
        r = evaluate(test, result, name, ref)
        final = result[0]
        r["_correct"] = test.correct[np.arange(len(final)), final].tolist()
        r["kind"] = kind
        return r

    policies = []
    for c in t.configs:
        r = ev(static(test, c), f"static {c}", "static")
        r["baseline"] = ", ".join(label[c]) or None
        policies.append(r)
    policies.append(ev(rules(test, mapping), "rules", "adaptive"))
    policies.append(ev(cascade(test, ladder, tau_star), f"cascade (tau={tau_star:.3f})", "adaptive"))
    policies.append(ev(cascade(test, ladder, tau_rc, start_test),
                       f"rules+cascade (tau={tau_rc:.3f})", "adaptive"))
    orc = ev(oracle(test), "oracle (upper bound, not deployable)", "oracle")
    policies.append(orc)
    deployable = [p for p in policies if p["kind"] != "oracle"]
    verdict = verdicts(deployable, names)
    for p in policies:
        p.pop("_correct", None)

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
        "cost_metric": t.cost_name, "cost_note": t.cost_note, "reference": ref, "ladder": ladder,
        "baselines": names, "tau": float(tau_star),
        "tau_rules_cascade": float(tau_rc), "rules": mapping, "n_calib": int(calib_m.sum()),
        "n_test": int(test_m.sum()), "policies": policies, "ablations": ablations,
        "frontier": [p["policy"] for p in pareto(deployable)],
        "escalation": {
            "cascade": escalation_analysis(test, ladder, tau_star),
            "rules+cascade": escalation_analysis(test, ladder, tau_rc, start_test),
        },
        "verdicts": verdict, "curve": curve, "curve_split": "held-out",
        "confidence": confidence_report(test), "confidence_split": "held-out",
        "per_task": per_task,
        "costs": dict(zip(t.configs, map(float, t.cost))),
    }
