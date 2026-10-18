import numpy as np

from foveate.prompts import mc_summary, parse_letter, softmax_over_options
from foveate.stats import auroc, bootstrap_ci, ece, paired_bootstrap_diff


def test_auroc():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5


def test_bootstrap_and_paired():
    x = np.r_[np.ones(70), np.zeros(30)]
    m, lo, hi = bootstrap_ci(x)
    assert m == 0.7 and lo < 0.7 < hi
    d, dlo, dhi = paired_bootstrap_diff(x, x)
    assert d == dlo == dhi == 0


def test_ece_perfect():
    assert ece([1.0, 1.0, 0.0], [1, 1, 0]) == 0.0


def test_option_softmax():
    p = softmax_over_options({"A": -0.1, "B": -2.5}, ["A", "B", "C"])
    assert abs(sum(p.values()) - 1) < 1e-9 and p["A"] > p["B"] > p["C"]
    s = mc_summary(p)
    assert s["pred"] == "A" and 0 < s["margin"] < 1
    assert parse_letter("(B) a red car", ["A", "B"]) == "B"
    assert parse_letter("none", ["A", "B"]) is None
