"""Send one gateway-style request straight to a vLLM server, then variants of it, and print
vLLM's answer to each. Pins down which request field a server rejects.

    python -m foveate.serve.probe --url http://localhost:8001 --model Qwen/Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import argparse

import httpx

from .. import config as C
from ..data import load_samples, synthetic
from ..data.frames import FrameStore
from ..prompts import question_text
from .gateway import sampled_fps, video_data_url


def build_body(model: str, url: str, text: str, fps: float) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": url}},
            {"type": "text", "text": text},
        ]}],
        "temperature": 0.0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20,
        "stream": True, "stream_options": {"include_usage": True},
        "media_io_kwargs": {"video": {"fps": fps}},
    }


def variants(body: dict) -> dict[str, dict]:
    drop = lambda *ks: {k: v for k, v in body.items() if k not in ks}  # noqa: E731
    return {
        "as sent by the gateway": body,
        "without media_io_kwargs": drop("media_io_kwargs"),
        "without logprobs": drop("logprobs", "top_logprobs"),
        "not streamed": drop("stream", "stream_options"),
        "plain (no logprobs, no media_io_kwargs, not streamed)":
            drop("logprobs", "top_logprobs", "media_io_kwargs", "stream", "stream_options"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--res", type=int, default=224)
    a = ap.parse_args(argv)
    synthetic.generate(C.data_root())
    s = load_samples("synthetic", "full")[0]
    fb = FrameStore(C.cache_root() / "probe", frame_counts={a.frames}, store_res=448).load(
        s, a.frames, a.res, None, False)
    body = build_body(a.model, video_data_url(fb), question_text(s), sampled_fps(fb))
    first_ok = None
    for name, b in variants(body).items():
        r = httpx.post(f"{a.url}/v1/chat/completions", json=b, timeout=300)
        print(f"[{r.status_code}] {name}: {' '.join(r.text.split())[:600]}")
        if r.status_code == 200 and first_ok is None:
            first_ok = name
    print("RESULT:", "every variant works" if first_ok == "as sent by the gateway"
          else f"first working variant: {first_ok}" if first_ok else "every variant was rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
