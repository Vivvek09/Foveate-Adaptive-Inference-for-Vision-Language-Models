"""End-to-end on CPU: grid -> replay -> shift -> report with a fake model."""

import json

from fake_backend import FakeBackend

from foveate.config import VisualConfig
from foveate.data import load_samples
from foveate.drift import shift_report
from foveate.grid import run_dir, run_grid
from foveate.policy import build_table, cascade, load_costs, load_rows, replay, static
from foveate.report import build_report


def test_end_to_end(tmp_env):
    samples = load_samples("synthetic", "full")
    cfgs = [VisualConfig.parse(s) for s in ["3b:2x224", "3b:4x448", "3b:8x448", "7b:8x672"]]
    fb = FakeBackend()
    for model in ("3b", "7b"):
        mine = [c for c in cfgs if c.model == model]
        for shard_id in range(2):  # two "GPUs"
            run_grid(fb, samples, mine, dataset="synthetic", subset="full", shard_id=shard_id,
                     num_shards=2, batch_size=5, probe=3, workers=2, store_res=672)
            run_grid(fb, samples, mine, dataset="synthetic", subset="full", corruption="blur:4",
                     shard_id=shard_id, num_shards=2, batch_size=5, probe=0, workers=2, store_res=672)
    # resumable: a second run adds nothing
    run_grid(fb, samples, cfgs[:1], dataset="synthetic", subset="full", shard_id=0, num_shards=2,
             batch_size=5, probe=0, store_res=672)
    rd = run_dir("synthetic", "full")
    df = load_rows(rd)
    assert len(df[df.corruption == "clean"]) == 24 * 4
    costs = load_costs(rd)
    t = build_table(df, costs)
    assert t.correct.shape == (24, 4) and (t.cost > 0).all()
    ladder = ["3b:2x224", "3b:8x448", "7b:8x672"]
    # tau=0 never escalates, tau>1 always runs every rung
    f0, c0, r0 = cascade(t, ladder, 0.0)
    assert (r0 == 1).all() and (f0 == t.col("3b:2x224")).all()
    f1, c1, r1 = cascade(t, ladder, 1.01)
    assert (r1 == 3).all()
    assert abs(c1.mean() - sum(t.cost[t.col(c)] for c in ladder)) < 1e-9
    assert static(t, "7b:8x672")[0][0] == t.col("7b:8x672")
    rep = replay(t, ladder)
    names = [p["policy"] for p in rep["policies"]]
    assert any(n.startswith("cascade") for n in names)
    assert "oracle (upper bound, not deployable)" in names
    assert "oracle (upper bound, not deployable)" not in rep["frontier"]
    assert rep["baselines"] == {"fixed cheapest": "3b:2x224", "static intermediate": "3b:8x448",
                                "fixed strongest": "7b:8x672"}
    # The fake backend's timings are mocked, so costs must fall back to model FLOPs.
    assert "TFLOPs" in rep["cost_metric"] and "mocked" in rep["cost_note"]
    esc = rep["escalation"]["cascade"]
    assert esc["helped"] + esc["hurt"] + esc["escalated_no_change_right"] \
        + esc["escalated_no_change_wrong"] == esc["escalated"]
    assert rep["verdicts"] and rep["curve_split"] == "held-out"
    out = rd / "replay"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(rep))
    srep = shift_report(df, costs, rep["ladder"], rep["tau"], {})
    (out / "shift.json").write_text(json.dumps(srep))
    assert srep["per_corruption"][0]["corruption"] == "blur:4"
    md = build_report(tmp_env / "results").read_text()
    assert "Stage 3: policies" in md and "Stage 4" in md and "MOCKED" in md
    assert list((tmp_env / "results" / "figures").glob("frontier_*.png"))
