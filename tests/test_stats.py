import numpy as np

from ckso import metrics as M
from ckso.stats import aggregate_seeds, holm, icc_2_1, paired_bootstrap


def test_basic_metrics():
    y = np.array([50.0, 60, 70, 40])
    p = np.array([52.0, 58, 70, 60])
    assert M.mae(y, p) == 6.0
    assert abs(M.rmse(y, p) - np.sqrt((4 + 4 + 0 + 400) / 4)) < 1e-12
    assert M.frac_abs_error_above(y, p, 15) == 0.25
    assert M.ef_metrics(y, y)["r2"] == 1.0
    assert np.isnan(M.mae([np.nan], [1.0]))


def test_octave_error_rate():
    assert M.octave_error_rate([60, 60, 60, 60], [120, 30, 61, 90]) == 0.5


def test_paired_bootstrap_detects_difference():
    rng = np.random.RandomState(0)
    y = rng.uniform(20, 80, 500)
    a = y + rng.randn(500) * 3
    b = y + rng.randn(500) * 6
    r = paired_bootstrap(y, a, b, n_boot=2000)
    assert r["mae"]["delta"] < 0 and r["mae"]["ci_high"] < 0 and r["mae"]["p"] < 0.01
    same = paired_bootstrap(y, a, a + 1e-9 * rng.randn(500), n_boot=2000)
    assert same["mae"]["ci_low"] <= 0 <= same["mae"]["ci_high"] or abs(same["mae"]["delta"]) < 1e-6


def test_holm():
    adj = holm([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.2])


def test_icc():
    x = np.random.RandomState(0).randn(50)
    assert abs(icc_2_1(np.stack([x, x], 1)) - 1.0) < 1e-12
    noisy = np.stack([x, x + np.random.RandomState(1).randn(50) * 3], 1)
    assert icc_2_1(noisy) < 0.5


def test_aggregate_seeds():
    agg = aggregate_seeds([{"ef": {"mae": 4.0}}, {"ef": {"mae": 5.0}}])
    assert agg["ef"]["mae"]["mean"] == 4.5 and abs(agg["ef"]["mae"]["std"] - np.sqrt(0.5)) < 1e-12


def test_binary_classification():
    out = M.binary_classification([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], threshold=0.3)
    assert abs(out["auc"] - 0.75) < 1e-12 and out["sensitivity"] == 1.0 and out["specificity"] == 0.5
