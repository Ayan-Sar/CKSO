"""Proposition 2, admissibility, Nyquist lock, closed-form powers and their gradients."""

import math

import pytest
import torch

from ckso.layout import LatentLayout
from ckso.operator import CKSOOperator, DenseOperator, omega_max, power_norms, rotation_matrix
from ckso.frequency import FrequencyHead

LAY = LatentLayout(5, 4, 3)


def random_op(lay=LAY, **kw):
    op = CKSOOperator(lay, **kw).double()
    with torch.no_grad():
        for p in op.parameters():
            p.add_(torch.randn_like(p))
    return op


def harm(omega1, H):
    return omega1[:, None] * torch.arange(1, H + 1, dtype=omega1.dtype)[None]


def test_dimensions_and_parameter_count():
    assert LAY.r == 22
    op = CKSOOperator(LAY)
    assert LAY.n_operator_scalars == 16
    assert sum(p.numel() for p in op.parameters()) == 16
    assert LatentLayout(3, 3, 2).r == 15
    dense = DenseOperator(LAY)
    assert sum(p.numel() for p in dense.parameters()) == 484


def test_initialisation_values():
    op = CKSOOperator(LAY)
    assert torch.allclose(op.rho_harm(), torch.full((5,), math.exp(-math.log1p(math.exp(-4)))))
    assert abs(float(op.rho_harm()[0]) - 0.982) < 1e-3
    assert torch.allclose(op.rho_perp(), torch.full((4,), 0.5))
    q = torch.sigmoid(op.b_perp)
    assert torch.allclose(q, torch.linspace(0.1, 0.9, 4), atol=1e-6)
    assert float(op.dc_eigenvalue()) == 1.0


@pytest.mark.parametrize("trial", range(5))
def test_admissibility_any_parameters(trial):
    op = random_op()
    with torch.no_grad():
        for p in op.parameters():
            p.mul_(10.0)  # extreme raw parameters
    w = torch.rand(3, dtype=torch.float64) * omega_max(5)
    K = op.matrix(harm(w, 5))
    eig = torch.linalg.eigvals(K).abs()
    assert torch.all(eig <= 1 + 1e-12)
    # DC eigenvalue is exactly one
    assert torch.all(K[:, 0, 0] == 1.0)


def test_normality_and_unit_norm_all_powers():
    op = random_op()
    w = torch.rand(4, dtype=torch.float64) * omega_max(5)
    K = op.matrix(harm(w, 5))
    assert torch.allclose(K.transpose(1, 2) @ K, K @ K.transpose(1, 2), atol=1e-12)
    sv = torch.linalg.svdvals(K)
    ev = torch.linalg.eigvals(K).abs()
    assert torch.allclose(torch.sort(sv, -1).values, torch.sort(ev, -1).values, atol=1e-10)
    norms = power_norms(K, 64)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)


def test_closed_form_powers_match_matrix_powers():
    op = random_op()
    B = 3
    w = torch.rand(B, dtype=torch.float64) * omega_max(5)
    hw = harm(w, 5)
    K = op.matrix(hw)
    z = torch.randn(B, 2, LAY.r, dtype=torch.float64)
    steps = [0, 1, 2, 7, 33, 64]
    out = op.propagate(z, steps, hw)
    for j, s in enumerate(steps):
        Ks = torch.linalg.matrix_power(K, s)
        ref = torch.einsum("bij,bnj->bni", Ks, z)
        assert torch.allclose(out[:, :, j], ref, atol=1e-9), s


def test_negative_scalar_modes_powers():
    lay = LatentLayout(1, 0, 2)
    op = CKSOOperator(lay).double()
    with torch.no_grad():
        op.c_scal.copy_(torch.tensor([-0.7, 0.3]))
    z = torch.randn(1, 1, lay.r, dtype=torch.float64)
    hw = torch.tensor([[0.3]], dtype=torch.float64)
    out = op.propagate(z, [1, 2, 3], hw)
    K = op.matrix(hw)[0]
    for j, s in enumerate([1, 2, 3]):
        assert torch.allclose(out[0, 0, j], torch.linalg.matrix_power(K, s) @ z[0, 0], atol=1e-12)


def test_lyapunov_non_increase():
    op = random_op()
    w = torch.rand(2, dtype=torch.float64) * omega_max(5)
    z = torch.randn(2, 10, LAY.r, dtype=torch.float64)
    Kz = op.step(z, harm(w, 5))
    assert torch.all((Kz ** 2).sum(-1) <= (z ** 2).sum(-1) + 1e-12)


def test_nyquist_safe_fundamental():
    for H in (1, 3, 4, 5, 9):
        head = FrequencyHead(r=10, H=H)
        u = torch.linspace(-50, 50, 1001)
        w1 = head.omega_max * torch.sigmoid(u)
        assert torch.all(H * w1 < math.pi)
    # equivalently H < fs / (2 f_HR) - 1/2: at 25 fps and 180 bpm, H <= 3
    fs, f_hr = 25.0, 3.0
    assert max(H for H in range(1, 20) if H < fs / (2 * f_hr) - 0.5) == 3


def test_frequency_head_initialises_at_detector():
    head = FrequencyHead(r=22, H=5)
    z = torch.randn(4, 30, 22)
    wh = torch.tensor([0.1, 0.2, 0.3, 0.5])
    assert torch.allclose(head(z, wh), wh, atol=1e-5)


def test_analytic_gradients_match_autograd():
    """d/drho and d/domega of sum_s <G_s, (rho R(omega))^s>_F match Proposition 2 (a)."""
    rho = torch.tensor(0.93, dtype=torch.float64, requires_grad=True)
    om = torch.tensor(0.41, dtype=torch.float64, requires_grad=True)
    S = 12
    G = torch.randn(S, 2, 2, dtype=torch.float64)
    L = sum((G[s - 1] * rho ** s * rotation_matrix(s * om)).sum() for s in range(1, S + 1))
    L.backward()
    J = rotation_matrix(torch.tensor(math.pi / 2, dtype=torch.float64))
    d_rho = sum(s * rho.detach() ** (s - 1) * (G[s - 1] * rotation_matrix(s * om.detach())).sum() for s in range(1, S + 1))
    d_om = sum(s * rho.detach() ** s * (G[s - 1] * (J @ rotation_matrix(s * om.detach()))).sum() for s in range(1, S + 1))
    assert torch.allclose(rho.grad, d_rho, atol=1e-10)
    assert torch.allclose(om.grad, d_om, atol=1e-10)


def test_harmonic_lock_chain_rule():
    """dL/domega_1 = sum_k k dL/domega_k through the lock."""
    op = random_op()
    w1 = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    k = torch.arange(1, 6, dtype=torch.float64)
    hw = w1[:, None] * k
    hw.retain_grad()
    z = torch.randn(1, 3, LAY.r, dtype=torch.float64)
    out = op.propagate(z, list(range(1, 20)), hw)
    (out ** 2 * torch.randn_like(out)).sum().backward()
    assert torch.allclose(w1.grad, (k * hw.grad[0]).sum(), atol=1e-10)


def test_spectral_radius_vs_transient_growth_example():
    A = torch.tensor([[0.9, 100.0], [0.0, 0.9]], dtype=torch.float64)
    assert abs(float(torch.linalg.eigvals(A).abs().max()) - 0.9) < 1e-12
    assert float(power_norms(A, 2)[1]) > 180


def test_nonnormal_variant_same_spectrum_transient_growth():
    op = CKSOOperator(LAY, normal=False, init_eta=3.0).double()
    w = torch.tensor([0.3], dtype=torch.float64)
    K = op.matrix(harm(w, 5))
    ref = CKSOOperator(LAY).double().matrix(harm(w, 5))
    ev = torch.sort(torch.linalg.eigvals(K).abs()).values
    ev_ref = torch.sort(torch.linalg.eigvals(ref).abs()).values
    assert torch.allclose(ev, ev_ref, atol=1e-8)
    assert float(power_norms(K, 64).max()) > 1.5
    # closed form still exact for the non-normal blocks
    z = torch.randn(1, 1, LAY.r, dtype=torch.float64)
    out = op.propagate(z, [5], harm(w, 5))[0, 0, 0]
    assert torch.allclose(out, torch.linalg.matrix_power(K[0], 5) @ z[0, 0], atol=1e-9)


def test_learned_dc_variant():
    op = CKSOOperator(LAY, learn_dc=True).double()
    with torch.no_grad():
        op.d_dc.fill_(0.01)
    w = torch.tensor([0.3], dtype=torch.float64)
    K = op.matrix(harm(w, 5))
    assert abs(float(power_norms(K, 1)[0]) - math.exp(0.01)) < 1e-10
    z = torch.randn(1, 1, LAY.r, dtype=torch.float64)
    assert torch.allclose(op.propagate(z, [3], harm(w, 5))[0, 0, 0],
                          torch.linalg.matrix_power(K[0], 3) @ z[0, 0], atol=1e-10)


def test_dense_operator_powers():
    op = DenseOperator(LAY).double()
    z = torch.randn(2, 3, LAY.r, dtype=torch.float64)
    hw = torch.zeros(2, 5, dtype=torch.float64)
    out = op.propagate(z, [0, 1, 4], hw)
    for j, s in enumerate([0, 1, 4]):
        ref = z @ torch.linalg.matrix_power(op.K, s).t()
        assert torch.allclose(out[:, :, j], ref, atol=1e-10)
