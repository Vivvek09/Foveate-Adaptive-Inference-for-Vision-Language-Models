"""Stage 5: a small policy-aware gateway in front of vLLM's OpenAI-compatible servers.

    client ──HTTP──> gateway (this file) ──> vllm serve 3B  (port 8001)
                     │ picks frames/res        └> vllm serve 7B-AWQ (port 8002)
                     │ runs the cascade
                     └ logs telemetry (JSONL)

The gateway owns the policy: it samples and resizes frames, sends them as a
`data:video/jpeg` URL (a comma-separated list of base64 JPEG frames, which
vLLM decodes without re-sampling), reads the option-letter log-probs, and
escalates to the next rung when confidence is below tau. vLLM owns batching,
KV-cache paging, prefix caching and the multimodal encoder cache.

Run:  foveate serve --dataset mvbench --subset pilot --backends 3b=http://localhost:8001 \
          7b=http://localhost:8002 --policy-file results/mvbench_pilot/replay/policy.json
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import VisualConfig, cache_root
from ..data import Sample
from ..data.frames import FrameBatch, FrameStore
from ..prompts import (
    clean_token,
    mc_summary,
    option_letters,
    parse_letter,
    question_text,
    softmax_over_options,
)

log = logging.getLogger(__name__)


def video_data_url(fb: FrameBatch, quality: int = 90) -> str:
    parts = []
    for im in fb.frames:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        parts.append(base64.b64encode(buf.getvalue()).decode())
    return "data:video/jpeg;base64," + ",".join(parts)


def sampled_fps(fb: FrameBatch) -> float:
    m = fb.meta
    dur = m.get("duration") or 0
    return len(fb.frames) / dur if dur else 1.0


class Gateway:
    def __init__(
        self,
        samples: dict[str, Sample],
        backends: dict[str, str],
        model_ids: dict[str, str],
        ladder: list[VisualConfig],
        tau: float,
        telemetry: Path,
        frame_workers: int = 4,
        store_res: int = 672,
    ):
        import httpx

        self.samples = samples
        self.backends = backends
        self.model_ids = model_ids
        self.ladder = ladder
        self.tau = tau
        self.telemetry = telemetry
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        self.store = FrameStore(cache_root() / f"r{store_res}",
                                frame_counts={c.frames for c in ladder}, store_res=store_res)
        self.pool = ThreadPoolExecutor(max_workers=frame_workers)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0),
                                        limits=httpx.Limits(max_connections=512))
        self._tlock = asyncio.Lock()

    async def _call(self, s: Sample, cfg: VisualConfig) -> dict:
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        fb = await loop.run_in_executor(self.pool, self.store.load, s, cfg.frames, cfg.res, None, False)
        url = await loop.run_in_executor(self.pool, video_data_url, fb)
        t_prep = time.perf_counter()
        body = {
            "model": self.model_ids[cfg.model],
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": url}},
                {"type": "text", "text": question_text(s)},
            ]}],
            "temperature": 0.0,
            "max_tokens": 1 if s.is_mc else 48,
            "logprobs": True,
            "top_logprobs": 20,
            "stream": True,
            "stream_options": {"include_usage": True},
            "media_io_kwargs": {"video": {"fps": sampled_fps(fb)}},
        }
        ttft = None
        text, lp_first, n_tok, usage = "", None, 0, {}
        async with self.client.stream("POST", f"{self.backends[cfg.model]}/v1/chat/completions",
                                      json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    delta = ch.get("delta", {}).get("content")
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t_prep
                        text += delta
                    lps = (ch.get("logprobs") or {}).get("content") or []
                    for item in lps:
                        n_tok += 1
                        if lp_first is None:
                            lp_first = item
        t_end = time.perf_counter()
        out = {"config": cfg.name, "prep_ms": (t_prep - t0) * 1e3,
               "ttft_ms": (ttft or (t_end - t_prep)) * 1e3, "backend_ms": (t_end - t_prep) * 1e3,
               "prompt_tokens": usage.get("prompt_tokens"), "text": text}
        if s.is_mc:
            letters = option_letters(s)
            letter_lp: dict[str, float] = {}
            for cand in (lp_first or {}).get("top_logprobs", []):
                L = clean_token(cand.get("token", ""))
                if L in letters:
                    letter_lp[L] = max(letter_lp.get(L, -math.inf), cand["logprob"])
            if letter_lp:
                out.update(mc_summary(softmax_over_options(letter_lp, letters)))
            else:
                out.update({"pred": parse_letter(text, letters), "conf": 0.0})
        else:
            out.update({"pred": text.strip(), "conf": 1.0})
        return out

    async def ask(self, qid: str, policy: str = "cascade", tau: float | None = None) -> dict:
        s = self.samples[qid]
        t0 = time.perf_counter()
        if policy == "cascade":
            rungs, thr = self.ladder, self.tau if tau is None else tau
        else:  # "static:3b:16x448"
            rungs, thr = [VisualConfig.parse(policy.split(":", 1)[1])], -1.0
        calls = []
        for cfg in rungs:
            res = await self._call(s, cfg)
            calls.append(res)
            if res.get("conf", 0.0) >= thr:
                break
        final = calls[-1]
        rec = {
            "qid": qid, "task": s.task, "policy": policy, "tau": thr,
            "pred": final.get("pred"), "answer": s.answer,
            "correct": (final.get("pred") == s.answer) if s.is_mc else None,
            "conf": final.get("conf"), "rungs": [c["config"] for c in calls],
            "ttft_ms": sum(c["prep_ms"] + c["backend_ms"] for c in calls[:-1])
            + calls[-1]["prep_ms"] + calls[-1]["ttft_ms"],
            "latency_ms": (time.perf_counter() - t0) * 1e3, "calls": calls,
        }
        async with self._tlock:
            with self.telemetry.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec


def build_app(gw: Gateway):
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="foveate gateway")

    @app.post("/ask")
    async def ask(
        qid: str = Body(...),
        policy: str = Body("cascade"),
        tau: float | None = Body(None),
    ):
        """Answer one question; policy is "cascade" or "static:MODEL:FxR"."""
        if qid not in gw.samples:
            raise HTTPException(404, f"unknown qid {qid}")
        return await gw.ask(qid, policy, tau)

    @app.get("/health")
    async def health():
        return {"ok": True, "ladder": [c.name for c in gw.ladder], "tau": gw.tau,
                "questions": len(gw.samples)}

    return app
