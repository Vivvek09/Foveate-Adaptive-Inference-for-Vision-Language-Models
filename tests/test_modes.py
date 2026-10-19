"""Smoke/pilot/full modes, T4 dtype rules, memory-fit skips, backend labelling (CPU only)."""

import json

import pytest
from fake_backend import FakeBackend

from foveate import config as C
from foveate.config import VisualConfig
from foveate.data import load_samples
from foveate.grid import run_dir, run_grid
from foveate.memory import check_fit, record_skips
from foveate.smoke import Checks, validate_grid


def test_modes_select_nested_subsets(tmp_env):
    cfg = C.load_config()
    smoke = load_samples("synthetic", "smoke")
    pilot = load_samples("synthetic", "pilot")
    full = load_samples("synthetic", "full")
    assert len(smoke) == 8 and len(full) == 24 and len(smoke) <= len(pilot) < len(full)
    assert {s.qid for s in smoke} <= {s.qid for s in pilot}  # smoke is drawn from the pilot
    assert {s.task for s in smoke} == {"color", "direction"}  # round-robin over tasks
    assert [s.qid for s in smoke] == [s.qid for s in load_samples("synthetic", "smoke")]  # fixed
    assert [c.name for c in C.grid_configs(cfg, "mvbench", "3b", "smoke")] == ["3b:4x224", "3b:8x448"]
    assert len(C.grid_configs(cfg, "mvbench", "3b", "pilot")) == 7


def test_resolve_dtype_never_silently_bf16_on_t4(monkeypatch):
    from foveate import utils

    monkeypatch.setattr(utils, "gpu_capability", lambda: (7, 5))
    monkeypatch.setattr(utils, "default_dtype", lambda: "float16")
    assert utils.resolve_dtype("auto") == "float16"
    assert utils.resolve_dtype("half") == "float16"
    with pytest.raises(ValueError, match="bf16"):
        utils.resolve_dtype("bfloat16")
    monkeypatch.setattr(utils, "gpu_capability", lambda: (8, 0))
    assert utils.resolve_dtype("bf16") == "bfloat16"


def test_memory_fit_on_one_t4():
    t4 = dict(total_gib=14.6)
    c3 = check_fit(VisualConfig.parse("3b:16x448"), "Qwen/Qwen2.5-VL-3B-Instruct", backend="vllm", **t4)
    c7 = check_fit(VisualConfig.parse("7b:16x448"), "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                   backend="vllm", **t4)
    assert c3.ok and c7.ok and 6 < c3.weights_gib < 8 and 5 < c7.weights_gib < 8
    # Unquantized 7B (15+ GiB of fp16 weights) cannot fit one T4: skipped, not substituted.
    big = check_fit(VisualConfig.parse("7b:8x448"), "Qwen/Qwen2.5-VL-7B-Instruct", backend="hf", **t4)
    assert not big.ok and "GiB" in big.reason
    # fp32 3B does not fit one T4 either (needs --tp 2).
    assert not check_fit(VisualConfig.parse("3b:8x448"), "Qwen/Qwen2.5-VL-3B-Instruct",
                         backend="vllm", dtype="float32", **t4).ok
    # 16 frames at 896 px overflows an 8192-token context.
    long = check_fit(VisualConfig.parse("3b:16x896"), "Qwen/Qwen2.5-VL-3B-Instruct", backend="vllm", **t4)
    assert not long.ok and "max_model_len" in long.reason


def test_skips_are_recorded_once(tmp_path):
    bad = check_fit(VisualConfig.parse("7b:8x448"), "Qwen/Qwen2.5-VL-7B-Instruct", backend="hf",
                    total_gib=14.6)
    record_skips(tmp_path, [bad])
    record_skips(tmp_path, [bad])
    lines = (tmp_path / "skipped.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["config"] == "7b:8x448"


class _OtherKind(FakeBackend):
    kind = "hf"


def test_grid_refuses_to_mix_backends(tmp_env):
    samples = load_samples("synthetic", "smoke")
    cfg = [VisualConfig.parse("3b:2x224")]
    run_grid(FakeBackend(), samples[:4], cfg, dataset="synthetic", subset="smoke", probe=0)
    with pytest.raises(RuntimeError, match="refusing"):
        run_grid(_OtherKind(), samples, cfg, dataset="synthetic", subset="smoke", probe=0)


def test_smoke_validation_rejects_mocked_timings(tmp_env):
    samples = load_samples("synthetic", "smoke")
    cfgs = [VisualConfig.parse("3b:2x224"), VisualConfig.parse("3b:8x448")]
    run_grid(FakeBackend(), samples, cfgs, dataset="synthetic", subset="smoke", probe=2,
             batch_size=4, store_res=448)
    ck = Checks()
    out = validate_grid(run_dir("synthetic", "smoke"), "3b", [c.name for c in cfgs], len(samples), ck,
                        synthetic=True)
    by = {(i["stage"], i["check"]): i["status"] for i in ck.items}
    assert out["rows"] == 16
    assert by[("preprocess", "visual tokens = formula")] == "PASS"
    assert by[("preprocess", "frame size multiple of 28")] == "PASS"
    assert by[("metrics", "no NaN confidences")] == "PASS"
    assert by[("storage", "replay reader loads rows")] == "PASS"
    assert by[("model load", "3b backend")] == "WARN"  # not vLLM
    # The fake backend's timings are not measurements, so the timing checks must fail.
    assert by[("timing", "3b:2x224 batch-1 latency")] == "FAIL"
    cost = json.loads(next((run_dir("synthetic", "smoke") / "cost").glob("*.json")).read_text())
    assert cost["measured"] is False and cost["backend"] == "fake"
