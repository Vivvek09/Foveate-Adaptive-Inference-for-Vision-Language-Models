"""Stage 5 on CPU: gateway + load generator against a fake OpenAI-compatible vLLM server."""

import asyncio
import json
import socket
import threading
import time

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")
httpx = pytest.importorskip("httpx")


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def fake_vllm(answer_conf: float):
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    app.state.bodies = []

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        app.state.bodies.append(body)
        url = body["messages"][0]["content"][0]["video_url"]["url"]
        n_frames = url.count(",") + 1
        import math

        lp_a = math.log(answer_conf)
        lp_b = math.log(1 - answer_conf)

        async def gen():
            chunk = {"choices": [{"index": 0, "delta": {"content": "A"}, "logprobs": {"content": [
                {"token": "A", "logprob": lp_a, "top_logprobs": [
                    {"token": "A", "logprob": lp_a}, {"token": "B", "logprob": lp_b}]}]}}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            yield f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 100 * n_frames}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _serve(app, port):
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(100):
        try:
            httpx.get(f"http://127.0.0.1:{port}/docs", timeout=0.5)
            return srv
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("server did not start")


def test_gateway_cascade_and_loadtest(tmp_env):
    from foveate.config import VisualConfig
    from foveate.data import load_samples
    from foveate.serve.gateway import Gateway, build_app
    from foveate.serve.loadgen import make_trace, run_trace

    samples = load_samples("synthetic", "full")
    p_small, p_big, p_gw = _port(), _port(), _port()
    small, big = fake_vllm(0.55), fake_vllm(0.95)  # cheap model unsure, big model sure
    _serve(small, p_small)
    _serve(big, p_big)
    ladder = [VisualConfig.parse("3b:2x224"), VisualConfig.parse("7b:4x448")]
    gw = Gateway({s.qid: s for s in samples},
                 {"3b": f"http://127.0.0.1:{p_small}", "7b": f"http://127.0.0.1:{p_big}"},
                 {"3b": "Qwen/Qwen2.5-VL-3B-Instruct", "7b": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"},
                 ladder, tau=0.8, telemetry=tmp_env / "results" / "serving" / "t.jsonl", store_res=448)
    _serve(build_app(gw), p_gw)
    url = f"http://127.0.0.1:{p_gw}"
    r = httpx.post(f"{url}/ask", json={"qid": samples[0].qid}, timeout=30)
    assert r.status_code == 200, r.text
    r = r.json()
    assert r["rungs"] == ["3b:2x224", "7b:4x448"] and r["conf"] > 0.8
    body = big.state.bodies[-1]
    assert body["model"] == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    assert body["messages"][0]["content"][0]["video_url"]["url"].startswith("data:video/jpeg;base64,")
    assert body["messages"][0]["content"][0]["video_url"]["url"].count(",") == 4  # 4 frames
    r = httpx.post(f"{url}/ask", json={"qid": samples[0].qid, "policy": "static:3b:2x224"},
                   timeout=30).json()
    assert r["rungs"] == ["3b:2x224"]
    trace = make_trace(samples, 12, rate=20.0, repeat=0.5)
    summ = asyncio.run(run_trace(url, trace, "cascade", tmp_env / "results" / "serving" / "x.jsonl"))
    assert summ["ok"] == 12 and summ["escalation_rate"] == 1.0 and summ["p95_ms"] > 0
