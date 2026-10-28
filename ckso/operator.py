"""Constrained Koopman Spectral Operator (Section III-A, III-B, Appendix C).

    K(omega_1, Theta) = [1] (+) (+)_k rho_k R(k omega_1) (+) (+)_p rho_p^perp R(omega_p^perp)
                        (+) (+)_q [varrho_q]

with rho = exp(-softplus(a)) in (0, 1], omega_perp = pi * sigmoid(b) in (0, pi),
varrho = tanh(c) in (-1, 1). Admissibility (all moduli <= 1) holds by construction.

Powers are closed form, K^s = [1] (+) (+)_k rho_k^s R(s k omega_1) (+) ..., so an
s-step prediction costs O(r) for any s and needs no backpropagation through time
(Proposition 2). The harmonic frequencies are passed in as ``harm_omegas`` of shape
(B, H): for CKSO proper these are k * omega_1 (harmonic lock); ablation variants pass
other values (free per-block frequencies, non-integer ratios).

Ablation operators implemented here:
  * ``learn_dc=True``        : DC eigenvalue lambda_0 = exp(d) learned (Table V row 5).
  * ``normal=False``         : harmonic blocks replaced by T_k (rho_k R(k omega_1)) T_k^{-1},
                               T_k = [[1, eta_k], [0, 1]]. Same eigenvalues as CKSO, but
                               non-normal for eta_k != 0, so ||K^s|| can exceed rho(K)^s
                               (Table V row 4). Powers remain closed form.
  * ``DenseOperator``        : unconstrained r x r matrix, powers by repeated
                               multiplication (BPTT) (Table V row 2).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .layout import LatentLayout


def omega_max(H: int) -> float:
    """Nyquist-safe upper bound pi / (H + 1/2): H * omega_1 < pi for any omega_1 < omega_max."""
    return math.pi / (H + 0.5)


def rotate(v: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Apply planar rotation R(angle) to v[..., 2]. ``angle`` broadcasts against v[..., 0]."""
    c, s = torch.cos(angle), torch.sin(angle)
    x, y = v[..., 0], v[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """R(angle) with shape angle.shape + (2, 2)."""
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def _as_steps(steps, device, dtype) -> torch.Tensor:
    if isinstance(steps, int):
        steps = [steps]
    t = torch.as_tensor(steps, device=device, dtype=dtype)
    if t.ndim != 1:
        raise ValueError("steps must be 1-D")
    if torch.any(t < 0) or torch.any(t != torch.round(t)):
        raise ValueError("steps must be non-negative integers")
    return t


class KoopmanOperatorBase(nn.Module):
    """Common interface for all latent transition operators.

    propagate(z, steps, harm_omegas) -> (B, N, S, r) where z is (B, N, r).
    matrix(harm_omegas)              -> (B, r, r)
    """

    closed_form: bool = True

    def __init__(self, layout: LatentLayout):
        super().__init__()
        self.layout = layout

    @property
    def r(self) -> int:
        return self.layout.r

    def propagate(self, z, steps, harm_omegas):  # pragma: no cover - abstract
        raise NotImplementedError

    def matrix(self, harm_omegas):  # pragma: no cover - abstract
        raise NotImplementedError

    def step(self, z: torch.Tensor, harm_omegas: torch.Tensor) -> torch.Tensor:
        """One application K z for z of shape (B, N, r)."""
        return self.propagate(z, [1], harm_omegas)[:, :, 0, :]

    def operator_scalars(self) -> list[nn.Parameter]:
        """Parameters of the operator (optimised with their own lr, no weight decay)."""
        return list(self.parameters())

    def spectrum(self, harm_omegas: torch.Tensor) -> torch.Tensor:
        """Eigenvalues (B, r) complex."""
        return torch.linalg.eigvals(self.matrix(harm_omegas).double())


class CKSOOperator(KoopmanOperatorBase):
    def __init__(
        self,
        layout: LatentLayout,
        learn_dc: bool = False,
        normal: bool = True,
        init_a_harm: float = -4.0,
        init_a_perp: float = 0.0,
        init_c_scal: float = 0.5,
        init_eta: float = 0.0,
        perp_angle_range: Sequence[float] = (0.1, 0.9),
    ):
        super().__init__(layout)
        H, P, Q = layout.H, layout.P, layout.Q
        self.learn_dc = learn_dc
        self.normal = normal
        # Harmonic radii rho_k = exp(-softplus(a_k)); a = -4 -> rho ~= 0.982 (Appendix D).
        self.a_harm = nn.Parameter(torch.full((H,), float(init_a_harm)))
        # Transverse radii (a = 0 -> rho = 0.5) and angles spread over sigma^{-1}{0.1..0.9}.
        self.a_perp = nn.Parameter(torch.full((P,), float(init_a_perp)))
        if P > 0:
            q = torch.linspace(perp_angle_range[0], perp_angle_range[1], P)
            b0 = torch.log(q) - torch.log1p(-q)
        else:
            b0 = torch.zeros(0)
        self.b_perp = nn.Parameter(b0)
        # Real scalar modes varrho_q = tanh(c_q). The manuscript does not state their
        # initialisation; c = 0.5 (varrho ~= 0.46) is our choice.
        self.c_scal = nn.Parameter(torch.full((Q,), float(init_c_scal)))
        if learn_dc:
            # lambda_0 = exp(d), d = 0 -> lambda_0 = 1. Not admissibility-constrained on
            # purpose (ablation row 5 reports a peak norm of 1.003).
            self.d_dc = nn.Parameter(torch.zeros(()))
        if not normal:
            self.eta = nn.Parameter(torch.full((H,), float(init_eta)))

    # ----- parameter maps ------------------------------------------------------
    def log_rho_harm(self) -> torch.Tensor:
        return -F.softplus(self.a_harm)

    def rho_harm(self) -> torch.Tensor:
        return torch.exp(self.log_rho_harm())

    def log_rho_perp(self) -> torch.Tensor:
        return -F.softplus(self.a_perp)

    def rho_perp(self) -> torch.Tensor:
        return torch.exp(self.log_rho_perp())

    def omega_perp(self) -> torch.Tensor:
        return math.pi * torch.sigmoid(self.b_perp)

    def varrho(self) -> torch.Tensor:
        return torch.tanh(self.c_scal)

    def dc_eigenvalue(self) -> torch.Tensor:
        if self.learn_dc:
            return torch.exp(self.d_dc)
        return torch.ones((), device=self.a_harm.device, dtype=self.a_harm.dtype)

    # ----- propagation -----------------------------------------------------------
    def propagate(self, z: torch.Tensor, steps, harm_omegas: torch.Tensor) -> torch.Tensor:
        """Closed-form K^s z for every s in ``steps``.

        z: (B, N, r); harm_omegas: (B, H) frequencies of the harmonic blocks;
        returns (B, N, S, r).
        """
        lay = self.layout
        B, N, _ = z.shape
        s = _as_steps(steps, z.device, z.dtype)
        S = s.shape[0]
        dc, harm, trans, scal = lay.split(z)

        # DC block: eigenvalue exactly 1 (a constant, not a parameter) unless ablated.
        if self.learn_dc:
            dc_s = dc[:, :, None, :] * torch.exp(s * self.d_dc)[None, None, :, None]
        else:
            dc_s = dc[:, :, None, :].expand(B, N, S, 1)

        # Harmonic blocks: rho_k^s R(s * omega_k).
        ang = s[None, :, None] * harm_omegas[:, None, :]  # (B, S, H)
        dec = torch.exp(s[:, None] * self.log_rho_harm()[None, :])  # (S, H)
        h = harm[:, :, None]  # (B, N, 1, H, 2)
        if self.normal:
            harm_s = dec[None, None, :, :, None] * rotate(h, ang[:, None])
        else:
            eta = self.eta  # T^{-1} h, rotate, T
            hx, hy = h[..., 0], h[..., 1]
            u = torch.stack([hx - eta * hy, hy], -1)
            u = dec[None, None, :, :, None] * rotate(u, ang[:, None])
            harm_s = torch.stack([u[..., 0] + eta * u[..., 1], u[..., 1]], -1)

        # Transverse blocks: rho_p^s R(s * omega_p^perp).
        if lay.P > 0:
            ang_p = s[:, None] * self.omega_perp()[None, :]  # (S, P)
            dec_p = torch.exp(s[:, None] * self.log_rho_perp()[None, :])
            trans_s = dec_p[None, None, :, :, None] * rotate(trans[:, :, None], ang_p)
        else:
            trans_s = trans.new_zeros(B, N, S, 0, 2)

        # Real scalar modes: varrho_q^s with sign handled explicitly.
        if lay.Q > 0:
            vr = self.varrho()
            mag = torch.pow(vr.abs()[None, :], s[:, None])  # (S, Q)
            odd = torch.remainder(s, 2.0)[:, None] == 1
            sign = torch.where(odd & (vr[None, :] < 0), -torch.ones_like(mag), torch.ones_like(mag))
            scal_s = scal[:, :, None, :] * (sign * mag)[None, None]
        else:
            scal_s = scal.new_zeros(B, N, S, 0)

        return lay.merge(dc_s, harm_s, trans_s, scal_s)

    # ----- explicit matrix -------------------------------------------------------
    def matrix(self, harm_omegas: torch.Tensor) -> torch.Tensor:
        """Explicit block-diagonal K (B, r, r), built independently of ``propagate``."""
        lay = self.layout
        B = harm_omegas.shape[0]
        mats = []
        rho_h = self.rho_harm()
        rho_p = self.rho_perp()
        om_p = self.omega_perp()
        vr = self.varrho()
        for b in range(B):
            blocks = [self.dc_eigenvalue().reshape(1, 1)]
            for k in range(lay.H):
                Bk = rho_h[k] * rotation_matrix(harm_omegas[b, k])
                if not self.normal:
                    T = torch.stack(
                        [torch.stack([torch.ones_like(self.eta[k]), self.eta[k]]),
                         torch.stack([torch.zeros_like(self.eta[k]), torch.ones_like(self.eta[k])])]
                    )
                    Tinv = torch.stack(
                        [torch.stack([torch.ones_like(self.eta[k]), -self.eta[k]]),
                         torch.stack([torch.zeros_like(self.eta[k]), torch.ones_like(self.eta[k])])]
                    )
                    Bk = T @ Bk @ Tinv
                blocks.append(Bk)
            for p in range(lay.P):
                blocks.append(rho_p[p] * rotation_matrix(om_p[p]))
            for q in range(lay.Q):
                blocks.append(vr[q].reshape(1, 1))
            mats.append(torch.block_diag(*blocks))
        return torch.stack(mats, 0)


class DenseOperator(KoopmanOperatorBase):
    """Unconstrained dense r x r transition (Table V row 2): r^2 parameters, BPTT powers."""

    closed_form = False

    def __init__(self, layout: LatentLayout, init_scale: float = 0.9, init_noise: float = 0.01, seed: int | None = None):
        super().__init__(layout)
        r = layout.r
        g = torch.Generator().manual_seed(0 if seed is None else seed)
        K0 = init_scale * torch.eye(r) + init_noise * torch.randn(r, r, generator=g) / math.sqrt(r)
        self.K = nn.Parameter(K0)

    def matrix(self, harm_omegas: torch.Tensor) -> torch.Tensor:
        B = harm_omegas.shape[0]
        return self.K[None].expand(B, -1, -1)

    def propagate(self, z: torch.Tensor, steps, harm_omegas: torch.Tensor) -> torch.Tensor:
        s = _as_steps(steps, z.device, torch.float64).long().tolist()
        smax = max(s)
        out = {0: z}
        cur = z
        for i in range(1, smax + 1):
            cur = cur @ self.K.t()
            out[i] = cur
        return torch.stack([out[i] for i in s], dim=2)


def build_operator(layout: LatentLayout, cfg: dict) -> KoopmanOperatorBase:
    kind = cfg.get("kind", "ckso")
    if kind == "ckso":
        return CKSOOperator(
            layout,
            learn_dc=bool(cfg.get("learn_dc", False)),
            normal=bool(cfg.get("normal", True)),
            init_a_harm=float(cfg.get("init_a_harm", -4.0)),
            init_a_perp=float(cfg.get("init_a_perp", 0.0)),
            init_c_scal=float(cfg.get("init_c_scal", 0.5)),
            init_eta=float(cfg.get("init_eta", 0.0)),
        )
    if kind == "dense":
        return DenseOperator(layout, init_scale=float(cfg.get("dense_init_scale", 0.9)),
                             init_noise=float(cfg.get("dense_init_noise", 0.01)))
    raise ValueError(f"Unknown operator kind '{kind}'")


@torch.no_grad()
def power_norms(K: torch.Tensor, max_power: int = 64) -> torch.Tensor:
    """||K^s||_2 for s = 1..max_power. K: (r, r) or (B, r, r). Computed in float64."""
    K = K.double()
    squeeze = K.ndim == 2
    if squeeze:
        K = K[None]
    P = torch.eye(K.shape[-1], dtype=K.dtype, device=K.device).expand_as(K).clone()
    norms = []
    for _ in range(max_power):
        P = P @ K
        norms.append(torch.linalg.matrix_norm(P, ord=2))
    out = torch.stack(norms, -1)
    return out[0] if squeeze else out
