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


def fake_vllm(answer_conf: float, served: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    app.state.bodies = []

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": served}]}

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
    # cheap model unsure, big model sure
    small, big = fake_vllm(0.55), fake_vllm(0.95, "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    _serve(small, p_small)
    _serve(big, p_big)
    ladder = [VisualConfig.parse("3b:2x224"), VisualConfig.parse("7b:4x448")]
    gw = Gateway({s.qid: s for s in samples},
                 {"3b": f"http://127.0.0.1:{p_small}", "7b": f"http://127.0.0.1:{p_big}"},
                 {"3b": "Qwen/Qwen2.5-VL-3B-Instruct", "7b": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"},
                 ladder, tau=0.8, telemetry=tmp_env / "results" / "serving" / "t.jsonl", store_res=448)
    _serve(build_app(gw), p_gw)
    url = f"http://127.0.0.1:{p_gw}"
    health = httpx.get(f"{url}/health", timeout=10).json()
    assert health["routing_ok"] is True and health["backends"]["7b"]["served"] == [
        "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"]
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
    assert trace == make_trace(samples, 12, rate=20.0, repeat=0.5)  # same load for every policy
    summ = asyncio.run(run_trace(url, trace, "cascade", tmp_env / "results" / "serving" / "x.jsonl",
                                 warmup=[samples[1].qid, samples[2].qid]))
    assert summ["ok"] == 12 and summ["escalation_rate"] == 1.0 and summ["p95_ms"] > 0
    assert summ["warmup_ok"] == 2 and summ["cold_first_ms"] > 0


def test_gateway_drops_rungs_without_backend_and_flags_wrong_model(tmp_env):
    from foveate.config import VisualConfig
    from foveate.data import load_samples
    from foveate.serve.gateway import Gateway, build_app

    samples = load_samples("synthetic", "full")
    p_small, p_gw = _port(), _port()
    _serve(fake_vllm(0.55, served="some/other-model"), p_small)
    ladder = [VisualConfig.parse("3b:2x224"), VisualConfig.parse("7b:4x448")]
    gw = Gateway({s.qid: s for s in samples}, {"3b": f"http://127.0.0.1:{p_small}"},
                 {"3b": "Qwen/Qwen2.5-VL-3B-Instruct", "7b": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"},
                 ladder, tau=0.8, telemetry=tmp_env / "t.jsonl", store_res=448)
    assert [c.name for c in gw.ladder] == ["3b:2x224"] and gw.dropped == ["7b:4x448"]
    _serve(build_app(gw), p_gw)
    url = f"http://127.0.0.1:{p_gw}"
    health = httpx.get(f"{url}/health", timeout=10).json()
    assert health["routing_ok"] is False and health["dropped_rungs"] == ["7b:4x448"]
    r = httpx.post(f"{url}/ask", json={"qid": samples[0].qid, "policy": "static:7b:4x448"}, timeout=30)
    assert r.status_code == 400


def test_gateway_reports_backend_rejection(tmp_env):
    """A vLLM 400 must reach the client with vLLM's own message, not as a bare 500."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from foveate.config import VisualConfig
    from foveate.data import load_samples
    from foveate.serve.gateway import Gateway, build_app

    bad = FastAPI()

    @bad.get("/v1/models")
    async def models():
        return {"data": [{"id": "Qwen/Qwen2.5-VL-3B-Instruct"}]}

    @bad.post("/v1/chat/completions")
    async def chat():
        return JSONResponse({"error": {"message": "video metadata missing fps"}}, status_code=400)

    samples = load_samples("synthetic", "full")
    p_small, p_gw = _port(), _port()
    _serve(bad, p_small)
    gw = Gateway({s.qid: s for s in samples}, {"3b": f"http://127.0.0.1:{p_small}"},
                 {"3b": "Qwen/Qwen2.5-VL-3B-Instruct"}, [VisualConfig.parse("3b:2x224")], tau=0.8,
                 telemetry=tmp_env / "t.jsonl", store_res=448)
    _serve(build_app(gw), p_gw)
    r = httpx.post(f"http://127.0.0.1:{p_gw}/ask", json={"qid": samples[0].qid}, timeout=30)
    assert r.status_code == 502
    assert "vLLM returned 400" in r.text and "video metadata missing fps" in r.text


def test_probe_reports_each_variant(tmp_env, capsys):
    from foveate.serve.probe import main

    p = _port()
    _serve(fake_vllm(0.9), p)
    main(["--url", f"http://127.0.0.1:{p}"])
    out = capsys.readouterr().out
    assert out.count("[200]") == 5 and "RESULT: every variant works" in out


def test_serve_scripts_allow_per_request_fps():
    # vLLM >= 0.32 rejects the gateway's media_io_kwargs (per-request fps) with a 400
    # unless the server is started with --trust-request-mm-kwargs (seen on Kaggle).
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("serve_and_load.sh", "serve_check.sh"):
        text = root.joinpath(name).read_text()
        cmds = re.findall(r"vllm serve .*?(?:\n(?!\s).*?$|2>&1)", text, flags=re.S | re.M)
        assert cmds, name
        for cmd in cmds:
            assert "--trust-request-mm-kwargs" in cmd, (name, cmd)
