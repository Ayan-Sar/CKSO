"""Theorem 3, H = 1 closed forms, exact root finding, soft-extremum bounds, EF gauge."""

import math

import numpy as np
import pytest
import torch

from ckso.layout import LatentLayout
from ckso.operator import CKSOOperator
from ckso.readout import GaugeFixedVolumeReadout
from ckso import functionals as fn


def setup(H=5, seed=0):
    torch.manual_seed(seed)
    lay = LatentLayout(H, 4, 3)
    op = CKSOOperator(lay).double()
    ro = GaugeFixedVolumeReadout(lay, vol_mean=80.0, vol_std=30.0).double()
    with torch.no_grad():
        ro.a.copy_(torch.randn(H))
        ro.w0.fill_(0.7)
        ro.b.fill_(0.3)
        for p in op.parameters():
            p.add_(torch.randn_like(p))
    return lay, op, ro


def test_theorem3_trace_equals_readout_of_propagated_state():
    lay, op, ro = setup()
    w1 = torch.tensor([0.37], dtype=torch.float64)
    hw = w1[:, None] * torch.arange(1, 6, dtype=torch.float64)
    z0 = torch.randn(1, 1, lay.r, dtype=torch.float64)
    steps = list(range(0, 40))
    V_direct = ro(op.propagate(z0, steps, hw))[0, 0]
    c0, alpha, phi = ro.coefficients(z0[:, 0])
    t = torch.tensor(steps, dtype=torch.float64)
    V_trig = fn.eval_trig_time(c0, alpha, phi, w1, t, rho=op.rho_harm())[0]
    assert torch.allclose(V_direct, V_trig, atol=1e-9)


def test_on_cycle_trace_is_periodic_trig_polynomial():
    lay, op, ro = setup()
    z0 = torch.randn(2, lay.r, dtype=torch.float64)
    c0, alpha, phi = ro.coefficients(z0)
    th = fn.phase_grid(128, dtype=torch.float64)
    V1 = fn.eval_trig_phase(c0, alpha, phi, th)
    V2 = fn.eval_trig_phase(c0, alpha, phi, th + 2 * math.pi)
    assert torch.allclose(V1, V2, atol=1e-9)
    assert torch.all(alpha >= 0)


def test_h1_closed_forms():
    c0 = torch.tensor([100.0], dtype=torch.float64)
    a = torch.tensor([30.0], dtype=torch.float64)
    phi = torch.tensor([0.8], dtype=torch.float64)
    ef, th_ed, th_es = fn.h1_closed_form(c0, a, phi)
    assert torch.allclose(ef, torch.tensor([100 * 2 * 30 / 130.0], dtype=torch.float64))
    ex = fn.exact_extrema(c0, a[:, None], phi[:, None])
    assert abs(float(ex["edv"][0]) - 130.0) < 1e-9 and abs(float(ex["esv"][0]) - 70.0) < 1e-9
    assert abs(float(ex["theta_ed"][0]) - float(th_ed[0])) < 1e-7
    assert abs(float(ex["theta_es"][0]) - float(th_es[0])) < 1e-7
    assert torch.allclose(fn.ejection_fraction(ex["edv"], ex["esv"]), ef)


@pytest.mark.parametrize("H", [1, 2, 3, 5, 8])
def test_exact_extrema_match_dense_grid(H):
    rng = np.random.RandomState(H)
    for _ in range(20):
        c0 = torch.tensor([rng.uniform(50, 150)], dtype=torch.float64)
        alpha = torch.tensor(rng.uniform(0, 20, (1, H)), dtype=torch.float64) / np.arange(1, H + 1)
        phi = torch.tensor(rng.uniform(0, 2 * np.pi, (1, H)), dtype=torch.float64)
        ex = fn.exact_extrema(c0, alpha, phi, safety_grid=4)
        th = fn.phase_grid(200000, dtype=torch.float64)
        V = fn.eval_trig_phase(c0, alpha, phi, th)[0]
        assert abs(float(ex["edv"][0]) - float(V.max())) < 1e-6
        assert abs(float(ex["esv"][0]) - float(V.min())) < 1e-6
        assert ex["n_roots"][0] >= 2


def test_critical_point_polynomial_roots_are_stationary():
    alpha = np.array([10.0, 4.0, 1.5])
    phi = np.array([0.3, 2.0, -1.0])
    th = fn.critical_phases(alpha, phi)
    k = np.arange(1, 4)
    dV = -(alpha[:, None] * k[:, None] * np.sin(k[:, None] * th[None] + phi[:, None])).sum(0)
    assert np.allclose(dV, 0, atol=1e-8)
    assert len(th) >= 2 and len(th) <= 6


def test_soft_extremum_bounds():
    torch.manual_seed(1)
    for beta in (2.0, 10.0, 40.0):
        for N in (16, 64, 256):
            c0 = torch.rand(50, dtype=torch.float64) * 100 + 50
            alpha = torch.rand(50, 5, dtype=torch.float64) * 10
            phi = torch.rand(50, 5, dtype=torch.float64) * 2 * math.pi
            th = fn.phase_grid(N, dtype=torch.float64)
            V = fn.eval_trig_phase(c0, alpha, phi, th)
            s = fn.soft_extrema(V, beta, th)
            vmax_grid = V.max(-1).values
            assert torch.all(s["edv"] <= vmax_grid + 1e-9)
            assert torch.all(s["edv"] >= vmax_grid - math.log(N) / beta - 1e-9)
            true_max = fn.exact_extrema(c0, alpha, phi)["edv"]
            bound = fn.soft_extremum_bound(N, beta, alpha)
            assert torch.all((s["edv"] - true_max).abs() <= bound + 1e-9)
            assert torch.all(s["esv"] >= V.min(-1).values - 1e-9)


def test_soft_phase_converges_to_argmax():
    c0 = torch.tensor([100.0], dtype=torch.float64)
    alpha = torch.tensor([[30.0, 5.0]], dtype=torch.float64)
    phi = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    th = fn.phase_grid(4096, dtype=torch.float64)
    V = fn.eval_trig_phase(c0, alpha, phi, th)
    s = fn.soft_extrema(V, 50.0, th)
    ex = fn.exact_extrema(c0, alpha, phi)
    d = (s["theta_ed"] - ex["theta_ed"] + math.pi) % (2 * math.pi) - math.pi
    assert abs(float(d)) < 0.02


def test_ef_gauge_scaling_invariant_offset_not():
    edv, esv = torch.tensor([120.0]), torch.tensor([50.0])
    ef = fn.ejection_fraction(edv, esv)
    assert torch.allclose(fn.ejection_fraction(3 * edv, 3 * esv), ef)
    assert not torch.allclose(fn.ejection_fraction(edv + 10, esv + 10), ef)


def test_readout_ignores_transverse_and_scalar_modes():
    lay, op, ro = setup()
    z = torch.randn(4, lay.r, dtype=torch.float64)
    z2 = z.clone()
    z2[:, lay.trans_slice] = torch.randn(4, 2 * lay.P, dtype=torch.float64)
    z2[:, lay.scal_slice] = torch.randn(4, lay.Q, dtype=torch.float64)
    assert torch.allclose(ro(z), ro(z2))
