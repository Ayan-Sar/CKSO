"""Linear volume observables V = w^T z + b (Section III-D, Appendix C.3).

``GaugeFixedVolumeReadout`` reads only the DC coordinate and the first coordinate of
each harmonic block, i.e. w_k = (a_k, 0). This removes the in-block rotation gauge and
makes the transverse/scalar modes invisible to V, which is the "vanishing transverse
modes" hypothesis of Theorem 3 imposed structurally on the readout.

Volumes are produced in physical units (mL):  V = sigma_V * (w^T z + b) + mu_V, with
(mu_V, sigma_V) training-split statistics (Appendix D). The affine standardisation only
improves conditioning; it does not change the function class.
"""

from __future__ import annotations

import torch
from torch import nn

from .layout import LatentLayout


class GaugeFixedVolumeReadout(nn.Module):
    def __init__(self, layout: LatentLayout, vol_mean: float = 0.0, vol_std: float = 1.0, init_a: float = 0.1):
        super().__init__()
        self.layout = layout
        self.w0 = nn.Parameter(torch.zeros(()))
        self.a = nn.Parameter(torch.full((layout.H,), float(init_a)))
        self.b = nn.Parameter(torch.zeros(()))
        self.register_buffer("vol_mean", torch.tensor(float(vol_mean)))
        self.register_buffer("vol_std", torch.tensor(float(vol_std)))

    def set_volume_stats(self, mean: float, std: float):
        self.vol_mean.fill_(float(mean))
        self.vol_std.fill_(float(max(std, 1e-6)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """V(z) in physical units for z (..., r)."""
        dc, harm, _, _ = self.layout.split(z)
        lin = self.w0 * dc[..., 0] + (harm[..., 0] * self.a).sum(-1) + self.b
        return self.vol_std * lin + self.vol_mean

    def coefficients(self, z: torch.Tensor):
        """Trigonometric coefficients of Theorem 3 for initial state z (..., r).

        V(t) = c0 + sum_k alpha_k rho_k^t cos(k omega_1 t + phi_k), with
        c0 = w0 z^(0) + b, alpha_k = ||w_k|| ||z^(k)||, phi_k = angle(z^(k)) - angle(w_k),
        all scaled to physical units.
        """
        dc, harm, _, _ = self.layout.split(z)
        c0 = self.vol_std * (self.w0 * dc[..., 0] + self.b) + self.vol_mean
        znorm = torch.sqrt(harm[..., 0] ** 2 + harm[..., 1] ** 2 + 1e-12)
        zang = torch.atan2(harm[..., 1], harm[..., 0])
        wang = torch.where(self.a >= 0, torch.zeros_like(self.a), torch.full_like(self.a, torch.pi))
        alpha = self.vol_std * self.a.abs() * znorm
        phi = zang - wang
        return c0, alpha, phi


class FullVolumeReadout(nn.Module):
    """Unrestricted linear readout over all r coordinates (used with dense / non-normal
    ablations, where Theorem 3 does not apply and the trace-based readout is used)."""

    def __init__(self, layout: LatentLayout, vol_mean: float = 0.0, vol_std: float = 1.0):
        super().__init__()
        self.layout = layout
        self.w = nn.Parameter(torch.zeros(layout.r))
        with torch.no_grad():
            self.w[layout.harm_slice][0::2] = 0.1
        self.b = nn.Parameter(torch.zeros(()))
        self.register_buffer("vol_mean", torch.tensor(float(vol_mean)))
        self.register_buffer("vol_std", torch.tensor(float(vol_std)))

    def set_volume_stats(self, mean: float, std: float):
        self.vol_mean.fill_(float(mean))
        self.vol_std.fill_(float(max(std, 1e-6)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.vol_std * (z @ self.w + self.b) + self.vol_mean
