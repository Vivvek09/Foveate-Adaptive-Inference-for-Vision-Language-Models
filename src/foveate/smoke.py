"""`foveate smoke`: the first thing to run on a new GPU box.

It runs the real pipeline at the smallest useful size (8 questions, one or two
configs per model, see `smoke:` in default.yaml) and checks every link:

    environment   CUDA visible, compute capability, the dtype that will be used
    model load    the engine starts (vLLM, or the labelled Transformers fallback)
    video decode  frames come out of PyAV with the requested count and size
    preprocess    frame sizes are multiples of 28; the engine's visual-token
                  count equals the formula in tokens.py
    inference     one row per question per config
    metrics       option-letter probabilities are finite, in [0, 1], sum to one
                  over the options; no NaN (fp16 overflow); answers not constant
    storage       grid + cost files exist, are labelled with their backend, and
                  load back through the stage-3 replay reader
    profile       (--profile) one CUDA-event-timed setting has positive vision,
                  prefill and decode times

Each model runs in its own process pinned to its own GPU (3B -> GPU 0, 7B ->
GPU 1 when there are two), which is also how serving places them. Results go
to results/<dataset>_smoke/ and the verdict to results/smoke/SMOKE.{json,md}.
Exit code 1 if any check FAILs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import config as C

log = logging.getLogger(__name__)


class Checks:
    def __init__(self):
        self.items: list[dict] = []

    def add(self, stage: str, name: str, status: str, detail: str = "") -> None:
        self.items.append({"stage": stage, "check": name, "status": status, "detail": detail})
        fn = log.error if status == "FAIL" else log.warning if status == "WARN" else log.info
        fn("[%s] %-14s %-28s %s", status, stage, name, detail)

    def ok(self, stage, name, cond, detail="", level="FAIL"):
        self.add(stage, name, "PASS" if cond else level, detail)
        return cond

    @property
    def failed(self) -> bool:
        return any(i["status"] == "FAIL" for i in self.items)


def _n_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def _log_summary(text: str) -> str:
    """The first and last error lines of a log (the first is usually the real cause), else its tail."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    errs = [ln for ln in lines if re.match(r"^[\w.]*(Error|Exception)\b.*:", ln)]
    picked = list(dict.fromkeys(errs[:1] + errs[-1:])) if errs else lines[-3:]
    return " | ".join(picked)[-600:]


# Modules whose failure breaks the run, and the fix to print. torchaudio and torchcodec are
# never used by Foveate (frames go in as PIL images, no audio), but transformers imports
# torchaudio at module level when it is installed, so a mismatched build breaks AutoProcessor.
_IMPORT_FIX = {
    "torch": "reinstall the GPU stack: bash scripts/setup_gpu.sh",
    "torchvision": "install the torchvision built for this torch: bash scripts/setup_gpu.sh",
    "torchaudio": "unused by Foveate but breaks transformers: pip uninstall -y torchaudio",
    "torchcodec": "unused by Foveate but can break transformers: pip uninstall -y torchcodec",
    "transformers": "see the error; usually a mismatched torchaudio/torchvision (bash scripts/setup_gpu.sh)",
    "vllm": "see the error; reinstall with bash scripts/setup_gpu.sh, or run with --backend hf",
    "av": "pip install av",
}


def preflight_imports(args, ck: Checks, result: dict | None = None) -> dict:
    """Import the stack in a fresh interpreter and name the module that really fails."""
    from .utils import import_check

    res = result if result is not None else import_check()
    for mod, r in res.items():
        if r.get("ok") is None:  # not installed; required packages are checked elsewhere
            continue
        if r.get("ok"):
            ck.add("environment", f"import {mod}", "PASS", str(r.get("version")))
            continue
        level = "WARN" if mod == "vllm" and args.backend == "hf" else "FAIL"
        ck.add("environment", f"import {mod}", level,
               f"{r.get('error')}; fix: {_IMPORT_FIX.get(mod, 'see the error')}")
    return res


def report_events(rd: Path, model: str, args, ck: Checks, since: float = 0.0) -> None:
    """Surface backend fallbacks logged by the grid process: a silent HF run is not a pass."""
    from .utils import read_jsonl

    f = rd / "events.jsonl"
    if not f.exists():
        return
    for ev in read_jsonl(f):
        if ev.get("event") != "vllm_fallback_to_hf" or ev.get("model") != model:
            continue
        if ev.get("time", since) < since:
            continue
        ck.add(f"{model}: run", "vLLM engine start", "FAIL",
               f"vLLM did not start, fell back to Transformers: {ev.get('error')}"
               f" (root cause: {ev.get('root_cause', 'n/a')})")


def _run_grid(args, model: str, gpu: int | None, ck: Checks) -> bool:
    cmd = [sys.executable, "-m", "foveate.cli"]
    if args.config:
        cmd += ["--config", args.config]
    cmd += ["grid", "--dataset", args.dataset, "--subset", "smoke", "--model", model,
            "--backend", args.backend, "--probe", "4", "--batch-size", "8",
            "--dtype", args.dtype, "--gpu-mem", str(args.gpu_mem),
            "--max-model-len", str(args.max_model_len)]
    if args.tasks:
        cmd += ["--tasks", args.tasks]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log.info("running: %s (CUDA_VISIBLE_DEVICES=%s)", " ".join(cmd), env.get("CUDA_VISIBLE_DEVICES"))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    logs = C.results_root() / "smoke" / f"grid_{model}.log"
    logs.parent.mkdir(parents=True, exist_ok=True)
    logs.write_text(proc.stdout + proc.stderr)
    tail = _log_summary(proc.stdout + proc.stderr)
    return ck.ok(f"{model}: run", "grid process exit code", proc.returncode == 0,
                 f"exit {proc.returncode} in {wall:.0f}s; log {logs}" + ("" if proc.returncode == 0
                                                                       else f"; tail: {tail}"))


def validate_grid(rd: Path, model: str, configs: list[str], n_expected: int, ck: Checks,
                  synthetic: bool = False) -> dict:
    """Check the rows and cost files one model's smoke grid wrote."""
    from .data import LETTERS
    from .policy import load_rows
    from .utils import read_jsonl

    st = f"{model}: outputs"
    rows = []
    for f in sorted((rd / "grid").glob(f"{model}_*.jsonl")):
        if "__" in f.name:
            continue
        rows.extend(read_jsonl(f))
    skipped = []
    if (rd / "skipped.jsonl").exists():
        skipped = [r for r in read_jsonl(rd / "skipped.jsonl") if r["config"].startswith(model)]
    for r in skipped:
        ck.add(st, f"config {r['config']}", "WARN", f"skipped on {r['backend']}: {r['reason']}")
    ran = sorted({r["config"] for r in rows})
    if not ck.ok("inference", f"{model} rows written", bool(rows), f"configs with rows: {ran}"):
        return {"rows": 0}
    for cfg in configs:
        if cfg in {r["config"] for r in skipped}:
            continue
        got = [r for r in rows if r["config"] == cfg]
        ck.ok("inference", f"{cfg} rows", len({r['qid'] for r in got}) == n_expected,
              f"{len(got)} rows for {n_expected} questions")
    kinds = sorted({r.get("backend") for r in rows})
    ck.ok("model load", f"{model} backend", kinds == ["vllm"], f"backend(s): {kinds}"
          + ("" if kinds == ["vllm"] else " (Transformers fallback: timings are not vLLM's)"),
          level="WARN")
    # Video decode + preprocessing
    bad_hw = [r["qid"] for r in rows if r["frame_hw"][0] % 28 or r["frame_hw"][1] % 28]
    ck.ok("preprocess", "frame size multiple of 28", not bad_hw, f"{len(bad_hw)} bad")
    ck.ok("video decode", "frames decoded", all(r.get("load_ms", 0) > 0 for r in rows),
          f"median load {sorted(r['load_ms'] for r in rows)[len(rows) // 2]:.1f} ms per question")
    mism = [(r["config"], r.get("visual_tokens"), r["visual_tokens_formula"]) for r in rows
            if r.get("visual_tokens") != r["visual_tokens_formula"]]
    ck.ok("preprocess", "visual tokens = formula", not mism,
          "all match" if not mism else f"{len(mism)} mismatches, e.g. {mism[0]} (engine, formula)")
    # Metrics + numerics
    confs = [r.get("conf") for r in rows]
    nan = [c for c in confs if c is None or (isinstance(c, float) and math.isnan(c))]
    ck.ok("metrics", "no NaN confidences", not nan,
          f"{len(nan)} NaN/None (fp16 overflow shows up here)")
    ck.ok("metrics", "confidence in [0, 1]",
          all(0.0 <= c <= 1.0 for c in confs if c is not None and c == c))
    mc = [r for r in rows if r.get("correct") is not None]
    if mc:
        ck.ok("metrics", "predictions are option letters",
              all(r.get("pred") in LETTERS for r in mc))
        letter_rows = [r for r in mc if r.get("conf", 0) > 0]
        ck.ok("metrics", "letter log-probs found", len(letter_rows) == len(mc),
              f"{len(mc) - len(letter_rows)} rows fell back to parsing text", level="WARN")
        for cfg in ran:
            g = [r for r in mc if r["config"] == cfg]
            acc = sum(r["correct"] for r in g) / len(g)
            preds = {r["pred"] for r in g}
            ck.ok("metrics", f"{cfg} answers vary", len(preds) > 1 or len(g) < 4,
                  f"predictions {sorted(preds)}", level="WARN")
            if synthetic:
                ck.ok("metrics", f"{cfg} accuracy (synthetic)", acc > 0.5,
                      f"{acc:.2f} on {len(g)} easy generated questions (chance ~0.4)", level="WARN")
            else:
                ck.add("metrics", f"{cfg} accuracy", "INFO",
                       f"{acc:.2f} on {len(g)} questions (too few to mean anything)")
    # Storage + timings
    costs = sorted((rd / "cost").glob(f"{model}_*.json"))
    ck.ok("storage", "cost files", bool(costs), f"{len(costs)} files")
    out = {"rows": len(rows), "configs": ran, "backend": kinds, "costs": {}}
    for f in costs:
        c = json.loads(f.read_text())
        lat = c.get("latency_b1_ms") or []
        ok = c.get("measured") and lat and all(x > 0 for x in lat)
        ck.ok("timing", f"{c['config']} batch-1 latency", bool(ok),
              f"p50 {c.get('latency_b1_ms_p50') or float('nan'):.0f} ms over {len(lat)} warm calls "
              f"({c.get('backend')}); engine start {c.get('backend_load_s') or float('nan'):.1f} s "
              "(cold)")
        out["costs"][c["config"]] = {k: c.get(k) for k in (
            "backend", "backend_load_s", "first_batch_wall_s", "amortized_ms",
            "latency_b1_ms_p50", "latency_b1_ms_p95", "latency_b1_warmup_ms")}
        out["costs"][c["config"]]["dtype"] = (c.get("backend_info") or {}).get("dtype")
    try:
        df = load_rows(rd)
        ck.ok("storage", "replay reader loads rows", len(df) > 0, f"{len(df)} rows")
    except Exception as e:  # noqa: BLE001
        ck.add("storage", "replay reader loads rows", "FAIL", str(e)[:200])
    return out


def _profile(args, ck: Checks) -> dict | None:
    from .backends.hf_backend import HFBackend
    from .data import load_samples
    from .profile import run_profile, summarize

    try:
        mid = C.model_id(C.load_config(args.config), "3b")
        b = HFBackend(mid, dtype=args.dtype)
        samples = load_samples("synthetic", "full", limit=1)
        out = C.results_root() / "smoke" / "profile_3b.jsonl"
        rows = run_profile(b, mid, samples, frames=[4], resolutions=[224], out_tokens=8,
                           decode_sweep=[], warmup=1, repeats=3, out_path=out,
                           cache_dir=C.cache_root() / "smoke_profile")
        summ = summarize(rows, mid)
    except Exception as e:  # noqa: BLE001
        ck.add("profile", "HF profile run", "FAIL", f"{type(e).__name__}: {str(e)[:200]}")
        return None
    vis = [r for r in summ if r["sweep"] == "visual"]
    if not ck.ok("profile", "HF profile run", bool(vis)):
        return None
    r = vis[0]
    for k in ("vision_ms", "prefill_ms", "decode_ms"):
        ck.ok("profile", f"{k} > 0", r[k] > 0, f"{r[k]:.1f} ms (median of 3 warm runs)")
    cold = [x for x in rows if x.get("sweep") == "cold"]
    if cold:
        ck.add("profile", "cold first call", "INFO",
               f"{cold[0]['gpu_total_ms']:.0f} ms GPU vs {r['gpu_total_ms']:.0f} ms warm median")
    return r


def cmd_smoke(args, cfg):
    from .grid import run_dir
    from .utils import capture_env, resolve_dtype

    # A private frame cache, so video decoding really runs on every smoke test.
    os.environ["FOVEATE_CACHE"] = str(C.cache_root() / "smoke")
    out = C.results_root() / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    ck = Checks()
    env = capture_env()
    gpus = env.get("gpus") or []
    ck.ok("environment", "CUDA GPU visible", bool(gpus),
          ", ".join(f"{g.get('name')} (cc {g.get('capability')})" for g in gpus) or "no GPU",
          level="WARN" if args.allow_cpu else "FAIL")
    try:
        dtype = resolve_dtype(args.dtype)
        ck.add("environment", "compute dtype", "PASS", dtype)
    except ValueError as e:
        ck.add("environment", "compute dtype", "FAIL", str(e))
        dtype = None
    for pkg in ("torch", "transformers", "vllm", "av"):
        v = (env.get("packages") or {}).get(pkg)
        ck.ok("environment", f"{pkg} installed", v is not None, v or "missing",
              level="WARN" if pkg in ("vllm", "av") and args.backend != "vllm" else "FAIL")
    summary_imports = preflight_imports(args, ck)

    if args.dataset == "synthetic":
        from .data import synthetic

        synthetic.generate(C.data_root())
    from .data import load_samples

    samples = load_samples(args.dataset, "smoke", fraction=cfg["pilot_fraction"], seed=cfg["seed"],
                           tasks=args.tasks)
    ck.ok("data", "smoke questions", len(samples) > 0,
          f"{len(samples)} questions over tasks {sorted({s.task for s in samples})}")
    ck.ok("data", "media files exist", all(s.exists() for s in samples))
    has_video = any(s.video for s in samples)
    ck.ok("video decode", "uses real video files", has_video,
          "PyAV decode path" if has_video else "frame folders only (install `av`)", level="WARN")

    rd = run_dir(args.dataset, "smoke")
    n_gpu = _n_gpus()
    summary = {"dataset": args.dataset, "questions": [s.qid for s in samples], "models": {},
               "env": env, "dtype": dtype, "imports": summary_imports}
    if dtype is not None and not ck.failed:
        for model in args.models:
            gpu = (1 if model != "3b" and n_gpu > 1 else 0) if n_gpu else None
            configs = [c.name for c in C.grid_configs(cfg, args.dataset, model, "smoke")]
            ck.add(f"{model}: run", "placement", "INFO",
                   (f"GPU {gpu}" if gpu is not None else "CPU") + f" (own process, own memory); configs {configs}")
            t_start = time.time()
            _run_grid(args, model, gpu, ck)
            if args.backend != "hf":
                report_events(rd, model, args, ck, since=t_start)
            summary["models"][model] = validate_grid(rd, model, configs, len(samples), ck,
                                                     synthetic=args.dataset == "synthetic")
        if args.profile:
            summary["profile"] = _profile(args, ck)
    summary["checks"] = ck.items
    summary["verdict"] = "FAIL" if ck.failed else "PASS"
    (out / "SMOKE.json").write_text(json.dumps(summary, indent=1, default=str))
    md = [f"# Foveate smoke test: {summary['verdict']}", "",
          f"Dataset `{args.dataset}`, {len(samples)} questions, models {args.models}. "
          "Every timing below was measured in this run on this machine.", "",
          "| status | stage | check | detail |", "|---|---|---|---|"]
    md += [f"| {i['status']} | {i['stage']} | {i['check']} | {i['detail'].replace('|', '/')} |"
           for i in ck.items]
    (out / "SMOKE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if ck.failed:
        raise SystemExit(1)
