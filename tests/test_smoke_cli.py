"""`foveate smoke` end to end on CPU: real subprocesses, real Transformers code path.

The model is a tiny *random* Qwen2.5-VL saved to disk and named in a config file,
so this checks the smoke runner's plumbing (process per model, grid, validation,
profile, SMOKE.md), not model quality and not GPU behaviour.
"""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("torchvision")
pytest.importorskip("transformers")
pytest.importorskip("av")


def test_smoke_command_with_tiny_model(tmp_env):
    import yaml

    from foveate.config import load_config

    import tiny_qwen

    model, proc = tiny_qwen.build()
    mdir = tmp_env / "tiny-qwen"
    model.save_pretrained(mdir)
    proc.save_pretrained(mdir)
    cfg = load_config()
    cfg["models"]["3b"]["id"] = str(mdir)
    cfg_path = tmp_env / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, "-m", "foveate.cli", "--config", str(cfg_path), "smoke", "--models", "3b",
         "--backend", "hf", "--dtype", "float32", "--allow-cpu", "--profile"],
        env=env, capture_output=True, text=True, timeout=900)
    res = tmp_env / "results"
    rep = json.loads((res / "smoke" / "SMOKE.json").read_text())
    status = {(c["stage"], c["check"]): c["status"] for c in rep["checks"]}
    assert r.returncode == 0, (r.stdout[-2000:], r.stderr[-2000:], status)
    assert rep["verdict"] == "PASS"
    assert status[("3b: run", "grid process exit code")] == "PASS"
    assert status[("inference", "3b:2x224 rows")] == "PASS"
    assert status[("inference", "3b:8x448 rows")] == "PASS"
    # The real processor's video-token expansion matches tokens.py.
    assert status[("preprocess", "visual tokens = formula")] == "PASS"
    assert status[("timing", "3b:2x224 batch-1 latency")] == "PASS"
    assert status[("model load", "3b backend")] == "WARN"  # hf, not vllm: flagged
    assert status[("profile", "vision_ms > 0")] == "PASS"
    assert (res / "smoke" / "SMOKE.md").read_text().startswith("# Foveate smoke test: PASS")
    cost = json.loads((res / "synthetic_smoke" / "cost" / "3b_2x224.s0of1.json").read_text())
    assert cost["backend"] == "hf" and cost["measured"] and cost["backend_info"]["dtype"] == "float32"
