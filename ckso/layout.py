"""Latent coordinate layout r = 1 + 2H + 2P + Q (Section III-A).

Ordering (fixed, part of the gauge fixing of Appendix C.3):

    index 0                      DC coordinate (eigenvalue exactly 1)
    1 .. 2H                      H harmonic 2-D blocks, block k turns by k*omega_1
    1+2H .. 2H+2P                P transverse 2-D blocks (rho_perp, omega_perp)
    1+2H+2P .. r-1               Q real scalar modes (varrho_q)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentLayout:
    H: int
    P: int
    Q: int

    def __post_init__(self):
        if self.H < 1:
            raise ValueError("H must be >= 1 (at least the fundamental block).")
        if self.P < 0 or self.Q < 0:
            raise ValueError("P and Q must be non-negative.")

    @property
    def r(self) -> int:
        return 1 + 2 * self.H + 2 * self.P + self.Q

    @property
    def n_operator_scalars(self) -> int:
        """Number of free operator scalars: H radii + P radii + P angles + Q reals."""
        return self.H + 2 * self.P + self.Q

    @property
    def harm_slice(self) -> slice:
        return slice(1, 1 + 2 * self.H)

    @property
    def trans_slice(self) -> slice:
        return slice(1 + 2 * self.H, 1 + 2 * self.H + 2 * self.P)

    @property
    def scal_slice(self) -> slice:
        return slice(1 + 2 * self.H + 2 * self.P, self.r)

    def split(self, z: torch.Tensor):
        """Split (..., r) into dc (..., 1), harm (..., H, 2), trans (..., P, 2), scal (..., Q)."""
        if z.shape[-1] != self.r:
            raise ValueError(f"Expected last dim {self.r}, got {z.shape[-1]}")
        lead = z.shape[:-1]
        dc = z[..., 0:1]
        harm = z[..., self.harm_slice].reshape(*lead, self.H, 2)
        trans = z[..., self.trans_slice].reshape(*lead, self.P, 2)
        scal = z[..., self.scal_slice]
        return dc, harm, trans, scal

    def merge(self, dc, harm, trans, scal) -> torch.Tensor:
        lead = dc.shape[:-1]
        return torch.cat(
            [
                dc,
                harm.reshape(*lead, 2 * self.H),
                trans.reshape(*lead, 2 * self.P),
                scal,
            ],
            dim=-1,
        )
