"""Analytic clinical functionals of the spectrum (Theorem 3, Section III-D, Appendix C.4).

Given trigonometric coefficients (c0, alpha_k, phi_k) of the on-cycle volume trace

    V(theta) = c0 + sum_{k=1}^H alpha_k cos(k theta + phi_k),   theta = omega_1 t  (mod 2 pi)

we provide
  * soft extrema on a *phase* grid (training; differentiable, with explicit error bound),
  * exact extrema via unit-circle roots of a degree-2H polynomial (inference),
  * EF = (EDV - ESV) / EDV (in percent) and ED/ES phases.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Re-exported so callers can obtain coefficients from a readout in one import.
from .readout import GaugeFixedVolumeReadout


def trig_coefficients(z: torch.Tensor, readout: GaugeFixedVolumeReadout):
    """(c0, alpha, phi) of Theorem 3 for latent states z (..., r)."""
    return readout.coefficients(z)


def phase_grid(N: int, device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.arange(N, device=device, dtype=dtype) * (2 * math.pi / N)


def eval_trig_phase(c0: torch.Tensor, alpha: torch.Tensor, phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """V at phases theta. c0 (...), alpha/phi (..., H), theta (N,) -> (..., N)."""
    H = alpha.shape[-1]
    k = torch.arange(1, H + 1, device=alpha.device, dtype=alpha.dtype)
    arg = k[:, None] * theta[None, :] + phi[..., :, None]  # (..., H, N)
    return c0[..., None] + (alpha[..., :, None] * torch.cos(arg)).sum(-2)


def eval_general_time(c0, alpha, phi, omegas, t: torch.Tensor) -> torch.Tensor:
    """V(t) = c0 + sum_k alpha_k cos(omega_k t + phi_k) for arbitrary (non-harmonic)
    block frequencies omegas (..., H); t (N,) -> (..., N). Used by free-frequency ablations."""
    arg = omegas[..., :, None] * t[None, :] + phi[..., :, None]
    return c0[..., None] + (alpha[..., :, None] * torch.cos(arg)).sum(-2)


def eval_trig_time(c0, alpha, phi, omega1, t: torch.Tensor, rho: torch.Tensor | None = None) -> torch.Tensor:
    """Full Theorem-3 trace including the decay rho_k^t. omega1 (...), t (N,)."""
    H = alpha.shape[-1]
    k = torch.arange(1, H + 1, device=alpha.device, dtype=alpha.dtype)
    arg = k[:, None] * omega1[..., None, None] * t[None, :] + phi[..., :, None]
    amp = alpha[..., :, None]
    if rho is not None:
        amp = amp * rho[:, None] ** t[None, :]
    return c0[..., None] + (amp * torch.cos(arg)).sum(-2)


def soft_extrema(V: torch.Tensor, beta: float, theta: torch.Tensor | None = None) -> dict:
    """Soft max/min over the last axis.

    EDV_beta = beta^{-1} log(N^{-1} sum_n exp(beta V_n)); ESV_beta symmetric with -beta.
    If ``theta`` (phase grid) is given, also returns circular soft-argmax phases.
    Bounds (Appendix C.4): V*_N - log(N)/beta <= EDV_beta <= V*_N.
    """
    N = V.shape[-1]
    logN = math.log(N)
    edv = (torch.logsumexp(beta * V, dim=-1) - logN) / beta
    esv = -(torch.logsumexp(-beta * V, dim=-1) - logN) / beta
    out = {"edv": edv, "esv": esv}
    if theta is not None:
        for name, sgn in (("theta_ed", 1.0), ("theta_es", -1.0)):
            p = torch.softmax(sgn * beta * V, dim=-1)
            s = (p * torch.sin(theta)).sum(-1)
            c = (p * torch.cos(theta)).sum(-1)
            out[name] = torch.remainder(torch.atan2(s, c), 2 * math.pi)
    return out


def soft_extremum_bound(N: int, beta: float, alpha: torch.Tensor) -> torch.Tensor:
    """Upper bound log(N)/beta + pi * sum_k k |alpha_k| / N on |EDV_beta - max V|."""
    H = alpha.shape[-1]
    k = torch.arange(1, H + 1, device=alpha.device, dtype=alpha.dtype)
    return math.log(N) / beta + math.pi * (k * alpha.abs()).sum(-1) / N


def ejection_fraction(edv: torch.Tensor, esv: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """EF in percent. EDV is clamped from below to keep the ratio finite before the
    positivity penalty has taken effect; with positive volumes this is the identity."""
    return 100.0 * (edv - esv) / torch.clamp(edv, min=eps)


def h1_closed_form(c0, alpha1, phi1):
    """Closed forms for H = 1: EF = 2 alpha / (c0 + alpha), theta_ED = -phi mod 2pi,
    theta_ES = (pi - phi) mod 2pi. Assumes alpha1 >= 0."""
    ef = 100.0 * 2 * alpha1 / (c0 + alpha1)
    th_ed = torch.remainder(-phi1, 2 * math.pi)
    th_es = torch.remainder(math.pi - phi1, 2 * math.pi)
    return ef, th_ed, th_es


def critical_phases(alpha: np.ndarray, phi: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Unit-circle roots of the degree-2H polynomial whose roots are the critical
    points of V(theta).

    With A_k = alpha_k e^{i phi_k} and nu = e^{i theta},
        V'(theta) = (1/2) sum_k (i k A_k nu^k - i k conj(A_k) nu^{-k}),
    so nu^H V'(theta) is the polynomial
        P(nu) = sum_k  i k A_k nu^{H+k}  -  i k conj(A_k) nu^{H-k}.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    H = alpha.shape[0]
    A = alpha * np.exp(1j * phi)
    coeffs = np.zeros(2 * H + 1, dtype=np.complex128)  # index = power of nu
    for k in range(1, H + 1):
        coeffs[H + k] += 1j * k * A[k - 1]
        coeffs[H - k] += -1j * k * np.conj(A[k - 1])
    poly = coeffs[::-1]  # highest power first
    scale = np.abs(poly).max()
    if scale == 0:
        return np.zeros(0)
    lead = 0
    while lead < len(poly) - 1 and abs(poly[lead]) < 1e-12 * scale:
        lead += 1
    poly = poly[lead:]
    if len(poly) < 2:
        return np.zeros(0)
    roots = np.roots(poly)
    on = np.abs(np.abs(roots) - 1.0) < tol
    return np.mod(np.angle(roots[on]), 2 * np.pi)


def _eval_np(c0, alpha, phi, theta):
    k = np.arange(1, alpha.shape[0] + 1)
    return c0 + (alpha[:, None] * np.cos(k[:, None] * theta[None, :] + phi[:, None])).sum(0)


def exact_extrema(c0: torch.Tensor, alpha: torch.Tensor, phi: torch.Tensor,
                  safety_grid: int = 64, tol: float = 1e-4) -> dict:
    """Exact EDV/ESV and ED/ES phases by root finding (non-differentiable; inference).

    Candidates are the unit-circle roots of the critical-point polynomial; a coarse
    phase grid is added as a numerical safety net (it can only be selected if root
    finding missed an extremum, e.g. a numerically degenerate leading coefficient).
    Inputs (B,), (B, H), (B, H); outputs are float64 tensors of shape (B,).
    """
    c0n = c0.detach().double().cpu().numpy().reshape(-1)
    an = alpha.detach().double().cpu().numpy().reshape(len(c0n), -1)
    pn = phi.detach().double().cpu().numpy().reshape(len(c0n), -1)
    grid = np.arange(safety_grid) * (2 * np.pi / safety_grid)
    res = {k: np.zeros(len(c0n)) for k in ("edv", "esv", "theta_ed", "theta_es", "n_roots")}
    for i in range(len(c0n)):
        roots = critical_phases(an[i], pn[i], tol=tol)
        cand = np.concatenate([roots, grid])
        v = _eval_np(c0n[i], an[i], pn[i], cand)
        imax, imin = int(np.argmax(v)), int(np.argmin(v))
        res["edv"][i], res["esv"][i] = v[imax], v[imin]
        res["theta_ed"][i], res["theta_es"][i] = cand[imax], cand[imin]
        res["n_roots"][i] = len(roots)
    return {k: torch.from_numpy(v) for k, v in res.items()}


def phase_to_frames(theta: torch.Tensor, omega1: torch.Tensor) -> torch.Tensor:
    """Frame offset after the reference time at which phase theta is reached."""
    return torch.remainder(theta, 2 * math.pi) / omega1
