"""Stage 1: where does the time go? A batch-1 sweep over frames x resolution.

For each setting the Transformers backend reports, per request:
    load (decode + resize, CPU) | preprocess (processor, CPU) | H2D copy |
    vision encoder | LLM prefill | decode (TPOT x tokens)
plus visual-token counts and peak memory. Every setting gets warm-up runs that
are discarded (first calls pay for CUDA context, cuBLAS heuristics and memory
allocation), then `repeats` timed runs; we keep medians and p95.

The second sweep fixes the visual input and varies the output length, which
separates the prefill cost (paid once) from the decode cost (paid per token).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import Sample
from .data.frames import FrameStore
from .tokens import decode_floor_seconds, gpu_spec, prefill_flops, shape_for, vit_patches
from .utils import capture_env, gpu_name, write_jsonl

log = logging.getLogger(__name__)


def _run_setting(backend, store, sample, n, res, out_tokens, warmup, repeats, extra,
                 record_cold: bool = False) -> list[dict]:
    fb = store.load(sample, n, res, with_stats=False)
    rows = []
    for w in range(warmup):
        t = backend.profile_one(sample, fb, out_tokens=out_tokens)
        if record_cold and w == 0:
            # The very first request of the process: CUDA context already exists
            # (model load), but kernels, cuBLAS heuristics and the allocator are cold.
            rows.append({"frames": n, "res": res, "repeat": -1, "frame_hw": [fb.height, fb.width],
                         "load_ms": fb.load_s * 1e3, **asdict(t), **extra, "sweep": "cold"})
    for r in range(repeats):
        fb = store.load(sample, n, res, with_stats=False)  # re-time the load as well
        t = backend.profile_one(sample, fb, out_tokens=out_tokens)
        rows.append({
            "frames": n, "res": res, "repeat": r, "frame_hw": [fb.height, fb.width],
            "load_ms": fb.load_s * 1e3, **asdict(t), **extra,
        })
    return rows


def run_profile(
    backend,
    model_id: str,
    samples: list[Sample],
    frames: list[int],
    resolutions: list[int],
    out_tokens: int,
    decode_sweep: list[int],
    warmup: int,
    repeats: int,
    out_path: Path,
    cache_dir: Path | None = None,
) -> list[dict]:
    store = FrameStore(cache_dir, frame_counts=frames, store_res=max(resolutions))
    sample = samples[0]
    env = capture_env()
    rows: list[dict] = []
    first = True
    for res in resolutions:
        for n in frames:
            fit = _fit(model_id, n, res)
            if fit is not None and not fit.ok:
                log.warning("frames=%d res=%d skipped: %s", n, res, fit.reason)
                rows.append({"frames": n, "res": res, "sweep": "visual", "skipped": fit.reason})
                continue
            try:
                rr = _run_setting(backend, store, sample, n, res, out_tokens, max(warmup, int(first)),
                                  repeats, {"sweep": "visual"}, record_cold=first)
                first = False
            except RuntimeError as e:  # most likely CUDA OOM at the largest settings
                log.warning("frames=%d res=%d failed: %s", n, res, str(e).splitlines()[0])
                _free_cuda()
                rows.append({"frames": n, "res": res, "sweep": "visual", "error": str(e)[:200]})
                continue
            rows.extend(rr)
            rr = [x for x in rr if x["sweep"] == "visual"]
            med = np.median([x["gpu_total_ms"] for x in rr])
            log.info("frames=%2d res=%3d tokens=%5d  total %.0f ms (vision %.0f, prefill %.0f, "
                     "decode %.0f)", n, res, rr[0]["visual_tokens"], med,
                     np.median([x["vision_ms"] for x in rr]),
                     np.median([x["prefill_ms"] for x in rr]),
                     np.median([x["decode_ms"] for x in rr]))
    # Decode sweep at a mid-sized input.
    n_mid, r_mid = sorted(frames)[len(frames) // 2], sorted(resolutions)[len(resolutions) // 2]
    for ot in decode_sweep or []:
        rows.extend(_run_setting(backend, store, sample, n_mid, r_mid, ot, warmup, repeats,
                                 {"sweep": "decode"}))
    for r in rows:
        r["model"] = model_id
        r["gpu"] = gpu_name()
    write_jsonl(out_path, rows)
    (out_path.with_suffix(".env.json")).write_text(__import__("json").dumps(env, indent=1))
    return rows


def _fit(model_id: str, frames: int, res: int):
    from .config import VisualConfig
    from .memory import check_fit

    try:
        return check_fit(VisualConfig("", frames, res), model_id, backend="hf")
    except Exception:  # noqa: BLE001 - the guard must never stop a profile
        return None


def _free_cuda():
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except ImportError:
        pass


def summarize(rows: list[dict], model_id: str, peak_tflops: float | None = None,
              mem_bw: float | None = None) -> list[dict]:
    """Median per setting, plus achieved TFLOP/s against the GPU's spec-sheet peak."""
    import pandas as pd

    df = pd.DataFrame([r for r in rows if "error" not in r and "skipped" not in r])
    if df.empty:
        return []
    spec = gpu_spec(df["gpu"].iloc[0]) or {}
    peak = peak_tflops or spec.get("tflops")
    bw = mem_bw or spec.get("mem_bw")
    shape = shape_for(model_id)
    out = []
    for (sweep, n, res, ot), g in df.groupby(["sweep", "frames", "res", "out_tokens"]):
        h, w = g["frame_hw"].iloc[0]
        rec = {
            "sweep": sweep, "frames": int(n), "res": int(res), "out_tokens": int(ot),
            "visual_tokens": int(g["visual_tokens"].iloc[0]),
            "prompt_tokens": int(g["prompt_tokens"].iloc[0]),
        }
        for col in ["load_ms", "preprocess_ms", "h2d_ms", "vision_ms", "prefill_ms",
                    "ttft_gpu_ms", "decode_ms", "tpot_ms", "gpu_total_ms", "peak_mem_gb"]:
            rec[col] = float(g[col].median())
        rec["gpu_total_p95_ms"] = float(g["gpu_total_ms"].quantile(0.95))
        rec["e2e_ms"] = rec["load_ms"] + rec["preprocess_ms"] + rec["gpu_total_ms"]
        if shape:
            fl = prefill_flops(shape, rec["prompt_tokens"], vit_patches(h, w, int(n)))
            rec["vision_tflops_achieved"] = fl["vit"] / (rec["vision_ms"] / 1e3) / 1e12
            rec["prefill_tflops_achieved"] = fl["llm"] / (rec["prefill_ms"] / 1e3) / 1e12
            if peak:
                rec["vision_mfu"] = rec["vision_tflops_achieved"] / peak
                rec["prefill_mfu"] = rec["prefill_tflops_achieved"] / peak
            if bw:
                rec["tpot_floor_ms"] = decode_floor_seconds(shape, bw) * 1e3
        out.append(rec)
    return out
