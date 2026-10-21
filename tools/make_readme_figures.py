"""Draw the README figures from docs/results/mvbench_2026-10-19.json.

    python tools/make_readme_figures.py

Writes docs/figures/*.png. Every plotted number comes from that JSON file, which is
copied from the GPU run's results (see its "about" field); nothing is typed in here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "results" / "mvbench_2026-10-19.json"
OUT = ROOT / "docs" / "figures"
ACCENT, GREY, DARK, LIGHT = "#2a6fdb", "#9a9a9a", "#222222", "#e6e6e6"


def _style(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=DARK, pad=22)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555555")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(colors="#444444", labelsize=9)
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.set_axisbelow(True)


def profile_figure(d: dict) -> None:
    rows = d["profile"]["rows"]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    for r in rows:
        gpu = r["vision_ms"] + r["prefill_ms"]
        big = r["visual_tokens"] >= 2000
        ax.scatter(r["visual_tokens"], gpu, s=36, color=ACCENT if big else GREY, zorder=3)
        if r["visual_tokens"] >= 4000:
            ax.annotate(f"{r['frames']} frames x {r['res']} px: {gpu:,.0f} ms",
                        (r["visual_tokens"], gpu), xytext=(-8, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=8.5, color=DARK)
    small = [r for r in rows if r["visual_tokens"] <= 1200]
    slope = sorted((r["vision_ms"] + r["prefill_ms"]) / r["visual_tokens"] for r in small)
    k = slope[len(slope) // 2]
    ax.plot([0, 4800], [0, 4800 * k], ls="--", color=GREY, lw=1, zorder=2)
    ax.text(2900, 2900 * k + 250, f"{k:.2f} ms per visual token", fontsize=8.5,
            color="#555555", ha="right")
    ax.set_xlim(0, 4800)
    ax.set_ylim(0, 9000)
    ax.set_xlabel("visual tokens", fontsize=9, color="#444444")
    ax.set_ylabel("GPU time: vision encoder + prefill (ms)", fontsize=9, color="#444444")
    _style(ax, "GPU time tracks visual tokens, and grows faster past ~2,000",
           "Qwen2.5-VL-3B on one T4, Transformers, batch 1, CUDA events; "
           "frames 1-16 x resolution 224-896 px")
    fig.tight_layout()
    fig.savefig(OUT / "profile_gpu_vs_tokens.png")
    plt.close(fig)


def frontier_figure(d: dict) -> None:
    rows = [r for r in d["mvbench_policies"]["rows"] if r["policy"] != "rules+cascade (tau=0.000)"]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    label = {"static 3b:4x224": ("3b:4x224", (0, -14)),
             "static 3b:8x448": ("3b:8x448", (0, 10)),
             "static 3b:16x448": ("3b:16x448", (0, -16)),
             "static 7b:16x448": ("7b:16x448 (reference)", (10, 0)),
             "rules": ("rules", (0, 12)),
             "cascade (tau=0.981)": ("cascade, tau 0.981", (-10, 0)),
             "oracle (upper bound, not deployable)": ("oracle (not deployable)", (10, 0))}
    for r in rows:
        adaptive = r["policy"] in ("rules", "cascade (tau=0.981)")
        oracle = r["policy"].startswith("oracle")
        color = ACCENT if adaptive else ("#555555" if oracle else GREY)
        ax.errorbar(r["mean_cost_ms"], r["accuracy"],
                    yerr=[[r["accuracy"] - r["acc_lo"]], [r["acc_hi"] - r["accuracy"]]],
                    fmt="o", color=color, ms=7 if adaptive or oracle else 5.5, elinewidth=1.1,
                    zorder=3)
        if r["policy"] in label:
            text, off = label[r["policy"]]
            ha = "left" if off[0] > 0 else ("right" if off[0] < 0 else "center")
            ax.annotate(text, (r["mean_cost_ms"], r["accuracy"]), xytext=off,
                        textcoords="offset points", ha=ha, va="center", fontsize=8.5,
                        color=DARK if adaptive or oracle else "#555555",
                        fontweight="bold" if adaptive else "normal")
    ax.set_xlim(0, 3100)
    ax.set_ylim(0.5, 0.9)
    ax.set_xlabel("mean cost per question: warm batch-1 latency on vLLM (ms), summed over rungs",
                  fontsize=9, color="#444444")
    ax.set_ylabel("accuracy (held-out)", fontsize=9, color="#444444")
    _style(ax, "Rules match the 7B within noise at 0.63x its latency; the cascade costs 1.73x",
           "MVBench, 567 held-out questions; whiskers are 95% bootstrap intervals")
    fig.tight_layout()
    fig.savefig(OUT / "mvbench_accuracy_vs_cost.png")
    plt.close(fig)


def serving_figure(d: dict) -> None:
    runs = d["serving"]["runs"]
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    for pol, color, name in (("cascade", ACCENT, "cascade"),
                             ("static:7b:16x448", GREY, "static 7b:16x448")):
        rs = sorted((r for r in runs if r["policy"] == pol), key=lambda r: r["rate"])
        x = [r["rate"] for r in rs]
        ax.plot(x, [r["p50_ms"] / 1000 for r in rs], "-o", color=color, lw=2.4, ms=6,
                label=f"{name}, p50")
        ax.plot(x, [r["p95_ms"] / 1000 for r in rs], "--o", color=color, lw=1.3, ms=4,
                label=f"{name}, p95")
        low = rs[0]
        ax.annotate(f"p50 {low['p50_ms'] / 1000:.1f} s", (low["rate"], low["p50_ms"] / 1000),
                    xytext=(10, -10 if pol == "cascade" else 6), textcoords="offset points",
                    fontsize=8.5, color=DARK if pol == "cascade" else "#555555",
                    fontweight="bold" if pol == "cascade" else "normal")
    ax.set_yscale("log")
    ax.set_xticks([0.5, 1, 2], ["0.5 req/s", "1 req/s", "2 req/s"])
    ax.set_yticks([5, 10, 20, 50, 100, 200, 400], ["5", "10", "20", "50", "100", "200", "400"])
    ax.set_xlim(0.35, 2.2)
    ax.set_xlabel("offered Poisson arrival rate", fontsize=9, color="#444444")
    ax.set_ylabel("latency at the client (s, log scale)", fontsize=9, color="#444444")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax, "Under load the cascade halves median latency at 0.5 req/s",
           "Live serving on 2x T4 (3B on GPU 0, 7B-AWQ on GPU 1), 200 requests per run, "
           "same trace for both policies")
    fig.tight_layout()
    fig.savefig(OUT / "serving_latency_vs_rate.png")
    plt.close(fig)


def main() -> None:
    d = json.loads(DATA.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    profile_figure(d)
    frontier_figure(d)
    serving_figure(d)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
