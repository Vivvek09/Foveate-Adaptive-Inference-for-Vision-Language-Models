"""Stage 2: run every question through every visual config once, and cache it.

Each row stores the answer, three confidence signals, token counts, and image
statistics. Every policy in stage 3 (static, rules, cascade, oracle) and the
drift study in stage 4 is then evaluated by *replaying* these rows, so trying a
new policy costs seconds instead of GPU hours. Cost per config is measured
twice: amortised throughput (batched GPU-seconds per question) and batch-1
latency from a small probe.

Cold vs warm: the cost file records the backend's start-up time (model load,
vLLM memory profiling and CUDA-graph capture), the first batch separately
(JIT/autotuning, encoder warm-up), the steady-state throughput over the
remaining batches, and batch-1 latency after discarded warm-up calls.

Every row and cost file names its backend ("vllm" or "hf"). A grid file is
never appended to by a different backend, so Transformers-fallback rows can
never be mixed into vLLM measurements.

Output layout (one directory per dataset+subset):
    results/<dataset>_<subset>/grid/<model>_<F>x<R>[__<corruption>].s<i>of<n>.jsonl
    results/<dataset>_<subset>/cost/<same stem>.json
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import VisualConfig, cache_root, results_root
from .data import Sample, shard
from .data.frames import FrameStore
from .tokens import visual_tokens
from .utils import capture_env, percentile, read_jsonl, write_jsonl

log = logging.getLogger(__name__)


def run_dir(dataset: str, subset: str) -> Path:
    return results_root() / f"{dataset}_{subset}"


def stem(cfg: VisualConfig, corruption: str | None) -> str:
    s = f"{cfg.model}_{cfg.frames}x{cfg.res}"
    if corruption:
        s += "__" + corruption.replace(":", "")
    return s


def _done_qids(path: Path, backend_kind: str | None = None) -> set[str]:
    if not path.exists():
        return set()
    rows = list(read_jsonl(path))
    kinds = {r.get("backend") for r in rows} - {None}
    if backend_kind and kinds and kinds != {backend_kind}:
        raise RuntimeError(
            f"{path} holds rows from backend {sorted(kinds)}, refusing to append {backend_kind!r} "
            "rows (timings and batching differ). Use another FOVEATE_RESULTS directory or "
            "delete the file.")
    return {r["qid"] for r in rows}


def run_grid(
    backend,
    samples: list[Sample],
    configs: list[VisualConfig],
    *,
    dataset: str,
    subset: str,
    corruption: str | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
    batch_size: int = 32,
    probe: int = 16,
    workers: int = 4,
    max_new_tokens: int = 48,
    store_res: int | None = None,
    probe_warmup: int = 2,
) -> None:
    """Run `configs` (all for one model) over this shard of `samples`, resumably."""
    if not configs:
        log.info("no configs to run")
        return
    out_dir = run_dir(dataset, subset)
    my = shard(samples, shard_id, num_shards)
    store_res = store_res or max(c.res for c in configs)
    # Cached frames are keyed by their stored resolution so a low-res run can
    # never feed upscaled frames to a later high-res run.
    store = FrameStore(cache_root() / f"r{store_res}", frame_counts={c.frames for c in configs},
                       store_res=store_res)
    env = capture_env()
    kind = getattr(backend, "kind", type(backend).__name__)
    binfo = backend.info() if hasattr(backend, "info") else {"kind": kind}
    pool = ThreadPoolExecutor(max_workers=workers)
    # Order configs by frame count so cached frames are reused immediately.
    for cfg in sorted(configs, key=lambda c: (c.frames, c.res)):
        name = stem(cfg, corruption)
        path = out_dir / "grid" / f"{name}.s{shard_id}of{num_shards}.jsonl"
        done = _done_qids(path, kind)
        todo = [s for s in my if s.qid not in done]
        log.info("[%s] %d to run (%d already done) on shard %d/%d",
                 name, len(todo), len(done), shard_id, num_shards)
        load = lambda s, c=cfg: store.load(s, c.frames, c.res, corruption)  # noqa: E731
        gen_wall = 0.0
        n_run = 0
        first_wall, first_n = None, 0
        chunks = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        nxt = pool.map(load, chunks[0]) if chunks else None
        t_cfg = time.perf_counter()
        for ci, chunk in enumerate(chunks):
            fbs = list(nxt)
            # Prefetch the next chunk's frames on CPU threads while the GPU works.
            nxt = pool.map(load, chunks[ci + 1]) if ci + 1 < len(chunks) else None
            preds, wall = backend.predict_batch(chunk, fbs, max_new_tokens=max_new_tokens)
            gen_wall += wall
            n_run += len(chunk)
            if ci == 0:
                first_wall, first_n = wall, len(chunk)
            rows = []
            for s, fb, p in zip(chunk, fbs, preds):
                row = {
                    "qid": s.qid, "dataset": s.dataset, "task": s.task, "clip_id": s.clip_id,
                    "question": s.question,
                    "config": cfg.name, "model": cfg.model, "frames": cfg.frames, "res": cfg.res,
                    "corruption": corruption, "answer": s.answer, "backend": kind,
                    "visual_tokens_formula": visual_tokens(fb.height, fb.width, cfg.frames),
                    "frame_hw": [fb.height, fb.width], "load_ms": fb.load_s * 1e3,
                    **{f"img_{k}": v for k, v in fb.stats.items()},
                    **p,
                }
                if s.is_mc:
                    row["correct"] = bool(p.get("pred") == s.answer)
                rows.append(row)
            write_jsonl(path, rows, append=True)
            if ci == 0 and chunk[0].is_mc:
                _sanity_check(rows, name)
            if (ci + 1) % 10 == 0:
                log.info("[%s] %d/%d", name, n_run, len(todo))
        # Batch-1 latency probe on the first few questions of the shard.
        lat, warm_lat = [], []
        if probe and my:
            probe_s = my[:probe]
            probe_fb = list(pool.map(load, probe_s))
            if probe_warmup:  # discarded: the first batch-1 calls pay for new shapes
                warm_lat = backend.latency_probe(probe_s[:probe_warmup], probe_fb[:probe_warmup],
                                                 max_new_tokens=max_new_tokens)
            lat = backend.latency_probe(probe_s, probe_fb, max_new_tokens=max_new_tokens)
        steady_n = n_run - first_n
        cost = {
            "config": cfg.name, "corruption": corruption, "shard": [shard_id, num_shards],
            "n_run": n_run, "gen_wall_s": gen_wall,
            "total_wall_s": time.perf_counter() - t_cfg,
            "amortized_ms": (gen_wall / n_run * 1e3) if n_run else None,
            "latency_b1_ms_p50": percentile(lat, 50) if lat else None,
            "latency_b1_ms_p95": percentile(lat, 95) if lat else None,
            "latency_b1_ms_p99": percentile(lat, 99) if lat else None,
            "latency_b1_ms": lat, "latency_b1_warmup_ms": warm_lat,
            "first_batch_wall_s": first_wall, "first_batch_n": first_n,
            "steady_amortized_ms": ((gen_wall - first_wall) / steady_n * 1e3)
            if steady_n > 0 and first_wall is not None else None,
            "backend_load_s": binfo.get("load_s"),
            "batch_size": batch_size, "env": env,
            "backend": kind, "backend_info": binfo,
            # Only real engines produce measurements; the test FakeBackend reports kind "fake".
            "measured": kind in ("vllm", "hf"),
        }
        cpath = out_dir / "cost" / f"{name}.s{shard_id}of{num_shards}.json"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        if n_run or not cpath.exists():
            cpath.write_text(json.dumps(cost, indent=1))
    pool.shutdown()


def _sanity_check(rows: list[dict], name: str) -> None:
    """fp16 overflow on GPUs without bf16 shows up as NaN or one constant answer."""
    confs = [r.get("conf") for r in rows]
    preds = [r.get("pred") for r in rows]
    if any(c is None or c != c for c in confs):
        log.warning("[%s] NaN confidences in the first batch: likely fp16 overflow. On T4 "
                    "try --dtype float32 --tp 2 (fp32 3B needs both GPUs)", name)
    elif len(rows) >= 16 and len(set(preds)) == 1:
        log.warning("[%s] every answer in the first batch is %r: check dtype/prompt", name, preds[0])
    acc = sum(r.get("correct", False) for r in rows) / max(1, len(rows))
    log.info("[%s] first-batch accuracy %.2f (chance is ~0.3 on MVBench)", name, acc)


class HFGridAdapter:
    """Lets the Transformers backend stand in for vLLM in run_grid (slow, batch 1).

    Its rows and costs are labelled backend="hf": accuracy is comparable with
    vLLM rows (same frames, same prompt, greedy), timings are not.
    """

    kind = "hf"

    def __init__(self, hf_backend):
        self.b = hf_backend

    def info(self) -> dict:
        return self.b.info()

    def predict_batch(self, samples, batches, max_new_tokens=48):
        t0 = time.perf_counter()
        rows = [self.b.predict(s, fb, max_new_tokens) for s, fb in zip(samples, batches)]
        return rows, time.perf_counter() - t0

    def latency_probe(self, samples, batches, max_new_tokens=48):
        # Processor + generate, like VLLMBackend.latency_probe (which includes vLLM's processor).
        out = []
        for s, fb in zip(samples, batches):
            t0 = time.perf_counter()
            self.b.predict(s, fb, max_new_tokens)
            out.append((time.perf_counter() - t0) * 1e3)
        return out
