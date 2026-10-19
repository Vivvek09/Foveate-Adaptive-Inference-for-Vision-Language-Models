"""Root-cause reporting for the import failures seen on Kaggle (AutoProcessor hiding torchaudio)."""
import json
from types import SimpleNamespace

from foveate.smoke import Checks, _log_summary, preflight_imports, report_events
from foveate.utils import import_check, root_cause


def _chained():
    try:
        try:
            raise OSError("libtorchaudio.so: undefined symbol: _ZN3c10")
        except OSError as inner:
            raise ModuleNotFoundError("Could not import module 'AutoProcessor'") from inner
    except ModuleNotFoundError as e:
        return e


def test_root_cause_walks_chain():
    assert root_cause(_chained()) == "OSError: libtorchaudio.so: undefined symbol: _ZN3c10"
    assert root_cause(ValueError("plain")) == "ValueError: plain"


def test_log_summary_prefers_first_and_last_error():
    log = "\n".join(["INFO loading", "OSError: undefined symbol in torchaudio",
                     "The above exception was the direct cause", "    ) from e",
                     "ModuleNotFoundError: Could not import module 'AutoProcessor'."])
    s = _log_summary(log)
    assert s.startswith("OSError: undefined symbol") and "AutoProcessor" in s
    assert _log_summary("a\nb\nc\nd") == "b | c | d"


def test_import_check_runs_in_subprocess():
    res = import_check(timeout=300)
    assert res["torch"]["ok"] is True
    assert set(res) >= {"torch", "transformers", "vllm"}


def test_preflight_names_broken_module():
    ck = Checks()
    fake = {"torch": {"ok": True, "version": "2.13.0"},
            "torchaudio": {"ok": False, "error": "OSError: undefined symbol"},
            "vllm": {"ok": None, "error": "not installed"}}
    preflight_imports(SimpleNamespace(backend="auto"), ck, result=fake)
    bad = [i for i in ck.items if i["status"] == "FAIL"]
    assert len(bad) == 1 and "pip uninstall -y torchaudio" in bad[0]["detail"]
    assert not any("vllm" in i["check"] for i in ck.items)


def test_fallback_event_is_a_failure(tmp_path):
    (tmp_path / "events.jsonl").write_text(json.dumps(
        {"time": 10.0, "event": "vllm_fallback_to_hf", "model": "3b", "error": "X",
         "root_cause": "OSError: torchaudio"}) + "\n")
    ck = Checks()
    report_events(tmp_path, "3b", SimpleNamespace(backend="auto"), ck, since=5.0)
    assert ck.failed and "torchaudio" in ck.items[0]["detail"]
    ck2 = Checks()
    report_events(tmp_path, "3b", SimpleNamespace(backend="auto"), ck2, since=20.0)  # old run
    assert not ck2.items
    (tmp_path / "events.jsonl").write_text(json.dumps(
        {"event": "vllm_fallback_to_hf", "model": "3b", "error": "X"}) + "\n")  # untimed, old format
    ck3 = Checks()
    report_events(tmp_path, "3b", SimpleNamespace(backend="auto"), ck3, since=5.0)
    assert not ck3.items


def test_report_lists_serving_errors(tmp_path):
    from foveate.report import build_report
    from foveate.serve.loadgen import summarize

    rows = [{"qid": "q", "ok": False, "error": "HTTPStatusError: 502 | body: video_url unsupported",
             "client_latency_ms": 1.0}] * 3
    summ = summarize(rows, 1.0, "cascade") | {"backend": "vllm serve", "rate": 1.0}
    assert summ["top_errors"][0][1] == 3
    (tmp_path / "serving").mkdir()
    (tmp_path / "serving" / "cascade_r1.0.summary.json").write_text(json.dumps(summ))
    md = build_report(tmp_path).read_text()
    assert "3/3 failed (3x)" in md and "video_url unsupported" in md
