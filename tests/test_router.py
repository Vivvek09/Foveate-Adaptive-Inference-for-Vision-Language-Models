"""Learned routing, temperature scaling and split robustness on a synthetic grid table."""

import numpy as np

from foveate.policy import Table, replay, static
from foveate.router import (
    apply_temperature,
    crossfit_probs,
    fit_router,
    fit_temperature,
    policy_family,
    predict,
    route,
    split_robustness,
)
from foveate.stats import ece

LADDER = ["3b:4x224", "3b:16x448", "7b:16x448"]


def table(n=800, seed=0):
    """Easy questions: every rung is right and the cheap rung is confident. Hard: only the top rung."""
    rng = np.random.default_rng(seed)
    hard = rng.random(n) < 0.4
    correct = np.stack([~hard | (rng.random(n) < 0.15), ~hard | (rng.random(n) < 0.5),
                        ~hard | (rng.random(n) < 0.9)], axis=1).astype(float)
    conf = np.clip(np.where(hard[:, None], 0.45, 0.9) + rng.normal(0, 0.08, (n, 3)), 0.26, 0.999)
    return Table(qids=np.array([f"q{i}" for i in range(n)]), tasks=np.array(["a", "b"] * (n // 2)),
                 questions=np.array(["how many cars pass?", "what happens next?"] * (n // 2)),
                 configs=LADDER, correct=correct, conf=conf, cost=np.array([100.0, 600.0, 1000.0]),
                 cost_name="latency_b1_ms")


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0.3, 0.9, 4000)
    correct = rng.random(4000) < true_p
    over = apply_temperature(true_p, 0.4)  # sharpened: overconfident
    t = fit_temperature(over, correct)
    assert t > 1.5
    assert ece(apply_temperature(over, t), correct) < ece(over, correct) / 2
    assert abs(fit_temperature(true_p, correct) - 1.0) < 0.25


def test_router_sends_hard_questions_up_and_keeps_easy_ones_cheap():
    t = table()
    r = fit_router(t, LADDER)
    assert predict(r, t).shape == (800, 3)
    final0, cost0, runs0 = route(t, r, 0.0)
    final, cost, runs = route(t, r, 0.2)
    acc = t.correct[np.arange(800), final].mean()
    cheap = t.correct[:, 0].mean()
    top = t.correct[:, 2].mean()
    assert acc > cheap + 0.2 and acc > top - 0.03
    assert cost.mean() < t.cost[2]            # cheaper than always running the top rung
    assert runs.max() <= 2 and (runs0 >= 1).all()
    # a huge lambda never leaves the probe
    f_inf, c_inf, r_inf = route(t, r, 1e6)
    assert (f_inf == 0).all() and (r_inf == 1).all() and np.allclose(c_inf, 100.0)
    assert np.isfinite(crossfit_probs(t, LADDER)).all()


def test_replay_reports_router_and_calibration_and_is_stable_across_splits():
    t = table()
    rep = replay(t, LADDER)
    names = [p["policy"] for p in rep["policies"]]
    assert any(n.startswith("router") for n in names)
    assert any(n.startswith("cascade, temperature-scaled") for n in names)
    assert {c["config"] for c in rep["calibration"]} == set(LADDER)
    assert abs(sum(rep["router"]["routes"].values()) - 1) < 1e-9
    router = next(p for p in rep["policies"] if p["policy"].startswith("router"))
    top = next(p for p in rep["policies"] if p["policy"] == "static 7b:16x448")
    assert router["mean_cost"] < top["mean_cost"] and router["accuracy"] > top["accuracy"] - 0.05
    assert static(t, LADDER[0])[1][0] == 100.0
    rob = split_robustness(t, LADDER, [0, 1, 2])
    fam = {r["policy"]: r for r in rob["policies"]}
    assert {"router", "cascade", "rules", "oracle"} <= set(fam)
    assert fam["router"]["n_splits"] == 3 and 0 <= fam["router"]["not_worse_share"] <= 1
    assert policy_family("cascade (tau=0.981)") == "cascade"


def test_goodput_is_the_highest_rate_that_meets_the_slo():
    from foveate.serve.loadgen import goodput, slo_attainment, summarize

    res = [{"ok": True, "client_latency_ms": 5_000.0}] * 9 + [{"ok": False, "error": "x",
                                                                "client_latency_ms": 1.0}]
    assert slo_attainment(res, 10_000) == 0.9       # a failure is a miss, however fast
    assert slo_attainment(res, 1_000) == 0.0
    s = summarize(res, wall=10.0, policy="cascade", slo_ms=10_000)
    assert s["slo_attainment"] == 0.9 and abs(s["goodput_rps"] - 0.9) < 1e-9
    runs = [{"policy": "cascade", "offered_rps": 0.5, "slo_attainment": 0.95},
            {"policy": "cascade", "offered_rps": 1.0, "slo_attainment": 0.91},
            {"policy": "cascade", "offered_rps": 2.0, "slo_attainment": 0.4},
            {"policy": "static", "offered_rps": 0.5, "slo_attainment": 0.6}]
    assert goodput(runs) == {"cascade": 1.0, "static": 0.0}
