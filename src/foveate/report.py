"""Figures and a Markdown summary from whatever results exist so far."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Categorical slots in fixed order (validated palette), plus recessive greys.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=INK2)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------


def plot_profile(summary: list[dict], out: Path) -> list[Path]:
    plt = _plt()
    vis = [r for r in summary if r["sweep"] == "visual"]
    paths = []
    if vis:
        vis = sorted(vis, key=lambda r: (r["res"], r["frames"]))
        labels = [f"{r['frames']}f @ {r['res']}" for r in vis]
        parts = [("load", "load_ms"), ("preprocess", "preprocess_ms"), ("vision encoder", "vision_ms"),
                 ("LLM prefill", "prefill_ms"), ("decode", "decode_ms")]
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(vis) + 1.6))
        left = np.zeros(len(vis))
        for k, (name, col) in enumerate(parts):
            vals = np.array([r.get(col, 0) or 0 for r in vis])
            ax.barh(labels, vals, left=left, color=C[k], label=name, height=0.7,
                    edgecolor="white", linewidth=1)
            left += vals
        _style(ax, "Where batch-1 latency goes (median ms)", "milliseconds", "")
        ax.legend(ncol=5, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
        p = out / "profile_breakdown.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        resolutions = sorted({r["res"] for r in vis})
        markers = ["o", "s", "^", "D", "v"]
        for k, res in enumerate(resolutions):
            pts = [r for r in vis if r["res"] == res]
            ax.plot([r["visual_tokens"] for r in pts], [r["ttft_gpu_ms"] for r in pts],
                    marker=markers[k % 5], color=C[k % 8], linewidth=1.5, markersize=6,
                    label=f"res {res}")
        x = np.array([r["visual_tokens"] for r in vis], float)
        y = np.array([r["ttft_gpu_ms"] for r in vis], float)
        if len(x) > 2:
            b, a = np.polyfit(x, y, 1)
            r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            xs = np.linspace(0, x.max(), 50)
            ax.plot(xs, a + b * xs, color=INK2, linestyle="--", linewidth=1,
                    label=f"fit: {b:.3f} ms/token, R²={r2:.3f}")
        _style(ax, "Time to first token grows with visual tokens", "visual tokens", "GPU TTFT (ms)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = out / "ttft_vs_tokens.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    dec = sorted([r for r in summary if r["sweep"] == "decode"], key=lambda r: r["out_tokens"])
    if dec:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ot = [r["out_tokens"] for r in dec]
        ax.plot(ot, [r["ttft_gpu_ms"] for r in dec], marker="o", color=C[0], linewidth=1.5,
                label="TTFT (prefill, paid once)")
        ax.plot(ot, [r["decode_ms"] for r in dec], marker="s", color=C[1], linewidth=1.5,
                label="decode (paid per token)")
        _style(ax, "Prefill vs decode as output length grows", "output tokens", "ms")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = out / "prefill_vs_decode.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_frontier(rep: dict, out: Path, tag: str) -> list[Path]:
    plt = _plt()
    paths = []
    fig, ax = plt.subplots(figsize=(7, 4.6))
    curve = rep["curve"]
    split = rep.get("curve_split", "all questions")
    ax.plot([c["mean_cost"] for c in curve], [c["accuracy"] for c in curve], color=C[0],
            linewidth=2, label=f"cascade, every threshold ({split})", zorder=2)
    for p in rep["policies"]:
        name = p["policy"]
        if name.startswith("static"):
            ax.scatter(p["mean_cost"], p["accuracy"], s=40, color="#8a8984", zorder=3,
                       edgecolor="white", linewidth=1.5)
            ax.annotate(name.replace("static ", ""), (p["mean_cost"], p["accuracy"]),
                        textcoords="offset points", xytext=(5, -10), fontsize=7, color=INK2)
    styles = {"cascade (": (C[0], "o"), "rules+cascade": (C[2], "D"), "rules": (C[1], "s"),
              "oracle": ("white", "*")}
    for p in rep["policies"]:
        for key, (col, m) in styles.items():
            if p["policy"].startswith(key):
                ax.scatter(p["mean_cost"], p["accuracy"], s=110 if m == "*" else 70, color=col,
                           marker=m, edgecolor=INK if m == "*" else "white", linewidth=1.2,
                           zorder=4, label=p["policy"])
                break
    _style(ax, f"Accuracy vs cost on held-out questions: {tag}",
           f"mean cost per question ({rep['cost_metric']})", "accuracy")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    p = out / f"frontier_{tag}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    pt = rep.get("per_task") or []
    if len(pt) > 1:
        cfgs = list(rep["costs"])
        mat = np.array([[r[c] for c in cfgs] for r in pt])
        fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(cfgs), 0.3 * len(pt) + 1.8))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cfgs)), cfgs, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pt)), [r["task"] for r in pt], fontsize=7)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] > 0.6 else INK)
        ax.set_title("Per-task accuracy by config: which tasks need frames, which need pixels",
                     loc="left", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        p = out / f"per_task_{tag}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_shift(rep: dict, out: Path, tag: str) -> list[Path]:
    plt = _plt()
    paths = []
    pc = rep["per_corruption"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(pc))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, [r["clean_escalation"] for r in pc], w, color=C[0], label="clean")
    ax.bar(x + w / 2 + 0.01, [r["shifted_escalation"] for r in pc], w, color=C[1], label="shifted")
    ax.set_xticks(x, [r["corruption"] for r in pc])
    _style(ax, "Cascade escalation rate, clean vs shifted inputs", "", "share of questions escalated")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = out / f"shift_escalation_{tag}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    for st in rep["streams"]:
        tr = st["trace"]
        n = len(tr["z"])
        k = max(5, st["window"] // 2)
        esc = np.convolve(tr["escalated"], np.ones(k) / k, mode="same")
        fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        axes[0].plot(range(n), tr["z"], color=C[0], linewidth=1.5, label="image-statistics drift (max |z|)")
        axes[0].axhline(st["z_alarm"], color=INK2, linestyle="--", linewidth=1, label="alarm threshold")
        _style(axes[0], f"Stream switches to {st['corruption']} at the dashed line", "", "drift z")
        axes[1].plot(range(n), esc, color=C[1], linewidth=1.5, label="escalation rate (moving avg)")
        _style(axes[1], "", "request index", "escalation rate")
        for a in axes:
            a.axvline(n // 2, color="#8a8984", linestyle=":", linewidth=1)
            a.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        p = out / f"shift_stream_{tag}_{st['corruption'].replace(':', '')}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(rows: list[dict], cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _cost_rows(run: Path) -> list[dict]:
    """Per config (merged over shards): backend, cold start, first batch, steady state, latency."""
    by: dict[str, dict] = {}
    for f in sorted((run / "cost").glob("*.json")):
        c = json.loads(f.read_text())
        key = c["config"] + ("" if not c.get("corruption") else f" [{c['corruption']}]")
        r = by.setdefault(key, {"config": key, "backend": set(), "measured": True, "lat": [],
                                "load": [], "first": [], "n": 0, "gen": 0.0})
        r["backend"].add(str(c.get("backend")))
        r["measured"] &= bool(c.get("measured", False))
        r["lat"] += c.get("latency_b1_ms") or []
        if c.get("backend_load_s") is not None:
            r["load"].append(c["backend_load_s"])
        if c.get("first_batch_wall_s") is not None:
            r["first"].append(c["first_batch_wall_s"])
        r["n"] += c.get("n_run") or 0
        r["gen"] += c.get("gen_wall_s") or 0.0
    out = []
    for r in by.values():
        lat = np.asarray(r["lat"], float)
        out.append({
            "config": r["config"], "backend": "+".join(sorted(r["backend"])),
            "source": "measured" if r["measured"] else "MOCKED (not a GPU result)",
            "questions": r["n"], "amortized_ms": r["gen"] / r["n"] * 1e3 if r["n"] else None,
            "engine_start_s": max(r["load"]) if r["load"] else None,
            "first_batch_s": max(r["first"]) if r["first"] else None,
            "b1_n": len(lat),
            "b1_p50_ms": float(np.percentile(lat, 50)) if len(lat) else None,
            "b1_p95_ms": float(np.percentile(lat, 95)) if len(lat) else None,
            "b1_p99_ms": float(np.percentile(lat, 99)) if len(lat) else None,
        })
    return out


def build_report(results: Path) -> Path:
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    md = ["# Foveate results", "", "Generated by `foveate report`. Every number below comes from "
          "result files in this folder; nothing is typed in by hand. Rows marked MOCKED come from "
          "the test stand-in model and are not GPU measurements.", ""]

    smoke = results / "smoke" / "SMOKE.md"
    if smoke.exists():
        first = smoke.read_text().splitlines()[0].lstrip("# ")
        md += ["## Smoke test", "", f"{first}. Details: [smoke/SMOKE.md](smoke/SMOKE.md).", ""]

    for p in sorted((results / "profile").glob("*.summary.json")):
        summ = json.loads(p.read_text())
        imgs = plot_profile(summ, figs)
        md += [f"## Stage 1: profile ({p.stem.replace('.summary', '')})", "",
               "Transformers backend, batch 1, CUDA events; medians of the timed repeats after "
               "discarded warm-up runs.", ""]
        md += [md_table([r for r in summ if r["sweep"] == "visual"],
                        ["frames", "res", "visual_tokens", "load_ms", "preprocess_ms", "vision_ms",
                         "prefill_ms", "decode_ms", "tpot_ms", "e2e_ms", "peak_mem_gb",
                         "prefill_mfu"]), ""]
        cold = [r for r in summ if r["sweep"] == "cold"]
        if cold:
            c = cold[0]
            warm = next((r for r in summ if r["sweep"] == "visual" and r["frames"] == c["frames"]
                         and r["res"] == c["res"]), None)
            md += [f"Cold vs warm: the first request of the process took {_fmt(c['gpu_total_ms'], 1)}"
                   f" ms of GPU time at {c['frames']} frames / {c['res']} px; the warm median at the "
                   f"same setting is {_fmt(warm['gpu_total_ms'] if warm else None, 1)} ms.", ""]
        skipped = [r for r in _read_jsonl(p.with_name(p.name.replace(".summary.json", ".jsonl")))
                   if "skipped" in r or "error" in r]
        if skipped:
            md += ["Settings not run:", ""]
            md += [md_table([{"frames": r["frames"], "res": r["res"],
                              "reason": r.get("skipped") or r.get("error")} for r in skipped],
                            ["frames", "res", "reason"]), ""]
        md += [f"![]({i.relative_to(results)})" for i in imgs] + [""]

    runs = sorted(d for d in results.iterdir() if d.is_dir() and ((d / "cost").exists()
                                                                  or (d / "replay").exists()))
    for run in runs:
        cr = _cost_rows(run)
        if cr:
            md += [f"## Stage 2: grid costs on {run.name}", "",
                   "engine_start_s is the cold start (weights, vLLM memory profiling, CUDA graphs); "
                   "first_batch_s includes kernel JIT and warm-up; amortized_ms is batched GPU time "
                   "per question; b1_* is warm batch-1 latency after discarded warm-up calls. "
                   "Transformers (hf) and vLLM timings are different systems and are not compared.",
                   ""]
            md += [md_table(cr, ["config", "backend", "source", "questions", "engine_start_s",
                                 "first_batch_s", "amortized_ms", "b1_n", "b1_p50_ms", "b1_p95_ms",
                                 "b1_p99_ms"]), ""]
        sk = _read_jsonl(run / "skipped.jsonl")
        if sk:
            md += ["Configs skipped (never substituted):", ""]
            md += [md_table(sk, ["config", "backend", "dtype", "reason"]), ""]
        ev = _read_jsonl(run / "events.jsonl")
        for e in ev:
            md += [f"- Event: `{e.get('event')}` for model {e.get('model')}: {e.get('error')}"]
        if ev:
            md += [""]
        rp = run / "replay" / "summary.json"
        if rp.exists():
            rep = json.loads(rp.read_text())
            imgs = plot_frontier(rep, figs, run.name)
            md += [f"## Stage 3: policies on {run.name}", "",
                   f"Cost metric: {rep['cost_metric']}."
                   + (f" Note: {rep['cost_note']}." if rep.get("cost_note") else "")
                   + f" Reference config: `{rep['reference']}` (chosen on the calibration half). "
                   f"Thresholds and rules fit on {rep['n_calib']} calibration questions, "
                   f"reported on {rep['n_test']} held-out questions. Ladder: "
                   + " → ".join(f"`{c}`" for c in rep["ladder"]) + ". The oracle peeks at the "
                   "answers and is an upper bound, not a deployable policy.", ""]
            md += [md_table(rep["policies"], ["policy", "baseline", "accuracy", "acc_lo", "acc_hi",
                                              "mean_cost", "cost_saving", "acc_diff", "acc_diff_lo",
                                              "acc_diff_hi", "escalation_rate"]), ""]
            if rep.get("verdicts"):
                md += ["Adaptive policies against the baselines (paired bootstrap on the same "
                       "held-out questions, 95% CI):", ""]
                md += [f"- {v}" for v in rep["verdicts"]] + [""]
            for name, e in (rep.get("escalation") or {}).items():
                md += [f"Escalation in {name}: {e['escalated']}/{e['n']} questions escalated; "
                       f"escalation fixed {e['helped']} and broke {e['hurt']} (net {e['net_gain']:+d}); "
                       f"accuracy {_fmt(e['accuracy_if_never_escalated'])} without escalation vs "
                       f"{_fmt(e['accuracy'])} with it, for {_fmt(e['extra_cost_from_escalation'], 2)} "
                       f"extra cost per question. {e['overconfident_wrong']} answers were accepted "
                       f"below the top rung with confidence ≥ tau and were wrong "
                       f"(mean confidence {_fmt(e['mean_conf_wrong_accepted'])}).", ""]
            md += ["Confidence quality per config on the held-out half (AUROC: does confidence "
                   "separate right from wrong answers? ECE: is it calibrated?):", ""]
            md += [md_table(rep["confidence"], ["config", "accuracy", "auroc", "ece", "cost"]), ""]
            if rep.get("ablations"):
                md += ["Ablations (cascade with one rung removed):", ""]
                md += [md_table(rep["ablations"], ["policy", "accuracy", "mean_cost", "acc_diff",
                                                   "cost_saving"]), ""]
            md += [f"Learned rules: `{json.dumps(rep['rules'])}`", ""]
            md += [f"![]({i.relative_to(results)})" for i in imgs] + [""]
        sp = run / "replay" / "shift.json"
        if sp.exists():
            rep = json.loads(sp.read_text())
            imgs = plot_shift(rep, figs, run.name)
            md += [f"## Stage 4: distribution shift on {run.name}", ""]
            md += [md_table(rep["per_corruption"],
                            ["corruption", "clean_cascade_acc", "shifted_cascade_acc",
                             "clean_escalation", "shifted_escalation", "clean_cost", "shifted_cost",
                             "clean_cheap_acc", "shifted_cheap_acc", "clean_cheap_auroc",
                             "shifted_cheap_auroc"]), ""]
            for st in rep["streams"]:
                md += [f"Stream with {st['corruption']}: alarm delay {st['detection_delay']} "
                       f"requests, {st['false_alarms_on_clean']} false alarms on clean. On the "
                       f"shifted half the drift-aware cascade cost "
                       f"{_fmt(st['drift_aware']['shifted']['mean_cost'])} vs "
                       f"{_fmt(st['plain']['shifted']['mean_cost'])} (plain), accuracy "
                       f"{_fmt(st['drift_aware']['shifted']['accuracy'])} vs "
                       f"{_fmt(st['plain']['shifted']['accuracy'])}.", ""]
            md += [f"![]({i.relative_to(results)})" for i in imgs] + [""]

    serving = sorted((results / "serving").glob("*.summary.json")) if (results / "serving").exists() else []
    if serving:
        rows = [json.loads(p.read_text()) | {"run": p.stem.replace(".summary", "")} for p in serving]
        md += ["## Stage 5: serving under load", "",
               "Open-loop Poisson arrivals; every policy at a given rate replays the same seeded "
               "trace. Latency is measured at the client. Warm-up requests are sent first and "
               "reported separately (cold_*), not mixed into the percentiles. p99 from a few "
               "hundred requests is noisy.", ""]
        md += [md_table(rows, ["run", "policy", "backend", "rate", "requests", "ok",
                               "throughput_rps", "p50_ms", "p95_ms", "p99_ms", "ttft_p50_ms",
                               "cold_p50_ms", "accuracy", "escalation_rate"]), ""]
        failed = [r for r in rows if r.get("ok", 0) < r.get("requests", 0)]
        if failed:
            md += ["Failed requests (most common error per run; full records in "
                   "`serving/*.jsonl`, server logs in `serving/vllm_*.log` and `gateway.log`):", ""]
            for r in failed:
                errs = r.get("top_errors") or [[r.get("warmup_first_error") or "not recorded", None]]
                err, n = errs[0][0], errs[0][1]
                md.append(f"- {r['run']}: {r['requests'] - r['ok']}/{r['requests']} failed"
                          + (f" ({n}x)" if n else "") + f": `{str(err).replace('`', '')[:300]}`")
            md.append("")
    out = results / "REPORT.md"
    out.write_text("\n".join(md))
    log.info("wrote %s", out)
    return out
