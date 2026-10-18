"""Open-loop load generator: Poisson arrivals against the gateway.

Open loop (arrivals do not wait for responses) is what real traffic looks like
and is the only way to see queueing: a closed-loop client slows down when the
server does, hiding exactly the tail latency we want to measure.

Workload knobs:
  rate        mean arrivals per second (Poisson, seeded)
  repeat      fraction of requests that reuse a recently seen clip with a
              different question (exercises vLLM's prefix + encoder caches)
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..data import Sample


def make_trace(samples: list[Sample], n: int, rate: float, repeat: float, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_clip: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_clip[s.clip_id or s.qid].append(s)
    recent: list[str] = []
    t = 0.0
    trace = []
    for _ in range(n):
        t += rng.expovariate(rate)
        reuse = [c for c in recent if len(by_clip[c]) > 1]
        if reuse and rng.random() < repeat:
            s = rng.choice(by_clip[rng.choice(reuse)])
        else:
            s = rng.choice(samples)
        recent = (recent + [s.clip_id or s.qid])[-32:]
        trace.append({"t": t, "qid": s.qid})
    return trace


async def run_trace(url: str, trace: list[dict], policy: str, out: Path, timeout: float = 600.0) -> dict:
    import httpx

    results = []
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=1024)) as client:
        t0 = time.perf_counter()

        async def one(item):
            delay = item["t"] - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            sent = time.perf_counter()
            try:
                r = await client.post(f"{url}/ask", json={"qid": item["qid"], "policy": policy})
                r.raise_for_status()
                rec = r.json()
                rec["client_latency_ms"] = (time.perf_counter() - sent) * 1e3
                rec["ok"] = True
            except Exception as e:  # keep going; failures are part of the result
                rec = {"qid": item["qid"], "ok": False, "error": str(e)[:200],
                       "client_latency_ms": (time.perf_counter() - sent) * 1e3}
            rec["sent_s"] = sent - t0
            results.append(rec)

        await asyncio.gather(*(one(it) for it in trace))
        wall = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in results:
            r.pop("calls", None)
            f.write(json.dumps(r) + "\n")
    return summarize(results, wall, policy)


def summarize(results: list[dict], wall: float, policy: str) -> dict:
    ok = [r for r in results if r.get("ok")]
    lat = np.array([r["client_latency_ms"] for r in ok]) if ok else np.array([np.nan])
    ttft = np.array([r.get("ttft_ms", np.nan) for r in ok]) if ok else np.array([np.nan])
    corr = [r["correct"] for r in ok if r.get("correct") is not None]
    esc = [len(r.get("rungs", [])) > 1 for r in ok]
    return {
        "policy": policy, "requests": len(results), "ok": len(ok), "wall_s": wall,
        "throughput_rps": len(ok) / wall if wall else float("nan"),
        "p50_ms": float(np.nanpercentile(lat, 50)), "p95_ms": float(np.nanpercentile(lat, 95)),
        "p99_ms": float(np.nanpercentile(lat, 99)), "mean_ms": float(np.nanmean(lat)),
        "ttft_p50_ms": float(np.nanpercentile(ttft, 50)), "ttft_p95_ms": float(np.nanpercentile(ttft, 95)),
        "accuracy": float(np.mean(corr)) if corr else None,
        "escalation_rate": float(np.mean(esc)) if esc else None,
    }
