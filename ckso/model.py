"""CKSO model: encoder + harmonic-lock operator + frequency head + analytic readout."""

from __future__ import annotations

import math

import torch
from torch import nn

from .layout import LatentLayout
from .networks import ConvEncoder, ConvDecoder, delay_embed
from .operator import build_operator, omega_max
from .frequency import FrequencyHead, FreeFrequencyHead
from .readout import GaugeFixedVolumeReadout, FullVolumeReadout
from . import functionals as fn

FREQUENCY_MODES = ("locked", "free_subject", "free_ratio")


class CKSOModel(nn.Module):
    """
    Variants (Table V) are selected by configuration:
      frequency_mode = 'locked'        CKSO: omega_k = k * omega_1             (rows 0, 4, 5, 6)
                       'free_subject'  per-block subject-conditioned omega_k   (row 1)
                       'free_ratio'    omega_k = kappa_k * omega_1, kappa_k free (row 3)
      operator.kind  = 'dense'         unconstrained r x r K                   (row 2)
      operator.normal = False          harmonic lock with non-normal blocks    (row 4)
      operator.learn_dc = True         DC eigenvalue learned                   (row 5)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        H, P, Q = int(cfg["H"]), int(cfg["P"]), int(cfg["Q"])
        self.layout = LatentLayout(H, P, Q)
        self.m = int(cfg.get("m", 4))
        self.image_size = int(cfg.get("image_size", 112))
        r = self.layout.r
        ch = tuple(cfg.get("encoder_channels", (32, 64, 96, 128)))
        bott = int(cfg.get("bottleneck", 32))
        groups = int(cfg.get("groups", 8))
        self.encoder = ConvEncoder(self.m, r, ch, bott, self.image_size, groups)
        self.decoder = ConvDecoder(r, 1, ch, bott, self.image_size, groups)
        op_cfg = dict(cfg.get("operator", {}))
        self.operator = build_operator(self.layout, op_cfg)

        self.frequency_mode = cfg.get("frequency_mode", "locked")
        if self.frequency_mode not in FREQUENCY_MODES:
            raise ValueError(f"frequency_mode must be one of {FREQUENCY_MODES}")
        hidden = int(cfg.get("freq_hidden", 64))
        self.freq_head = FrequencyHead(r, H, hidden)
        if self.frequency_mode == "free_subject":
            self.free_head = FreeFrequencyHead(r, H, hidden)
        if self.frequency_mode == "free_ratio":
            k = torch.arange(1, H + 1, dtype=torch.float32) / (H + 0.5)
            self.kappa_logit = nn.Parameter(torch.log(k) - torch.log1p(-k))

        is_normal = op_cfg.get("kind", "ckso") == "ckso" and bool(op_cfg.get("normal", True))
        mode = cfg.get("readout_mode", "auto")
        if mode == "auto":
            if not is_normal:
                mode = "trace"
            elif self.frequency_mode == "locked":
                mode = "analytic"
            else:
                mode = "general"
        self.readout_mode = mode
        if mode == "trace":
            self.readout = FullVolumeReadout(self.layout)
        else:
            self.readout = GaugeFixedVolumeReadout(self.layout)
        self.n_phase_grid = int(cfg.get("n_phase_grid", 64))
        self.trace_frames = int(cfg.get("trace_frames", 64))

    # ------------------------------------------------------------------ utilities
    @property
    def r(self) -> int:
        return self.layout.r

    @property
    def omega_max(self) -> float:
        return omega_max(self.layout.H)

    def set_volume_stats(self, mean: float, std: float):
        self.readout.set_volume_stats(mean, std)

    def operator_parameters(self) -> list[nn.Parameter]:
        ps = list(self.operator.parameters())
        if self.frequency_mode == "free_ratio":
            ps.append(self.kappa_logit)
        return ps

    def inference_modules(self) -> list[nn.Module]:
        """Modules needed at inference (the decoder is training-only)."""
        mods = [self.encoder, self.operator, self.freq_head, self.readout]
        if self.frequency_mode == "free_subject":
            mods.append(self.free_head)
        return mods

    def count_parameters(self, inference_only: bool = True) -> int:
        if not inference_only:
            return sum(p.numel() for p in self.parameters())
        n = sum(p.numel() for mod in self.inference_modules() for p in mod.parameters())
        if self.frequency_mode == "free_ratio":
            n += self.kappa_logit.numel()
        return n

    # ------------------------------------------------------------------ encoder / decoder
    def encode(self, frames: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """frames (B, T, H, W) -> z (B, T-m+1, r)."""
        X = delay_embed(frames, self.m)  # (B, T', m, H, W)
        B, Tp = X.shape[:2]
        flat = X.reshape(B * Tp, *X.shape[2:])
        if chunk is None or flat.shape[0] <= chunk:
            z = self.encoder(flat)
        else:
            z = torch.cat([self.encoder(flat[i:i + chunk]) for i in range(0, flat.shape[0], chunk)], 0)
        return z.view(B, Tp, self.r)

    def decode(self, z: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        lead = z.shape[:-1]
        flat = z.reshape(-1, self.r)
        if chunk is None or flat.shape[0] <= chunk:
            x = self.decoder(flat)
        else:
            x = torch.cat([self.decoder(flat[i:i + chunk]) for i in range(0, flat.shape[0], chunk)], 0)
        return x.view(*lead, self.image_size, self.image_size)

    # ------------------------------------------------------------------ frequencies
    def frequencies(self, z: torch.Tensor, omega_hat: torch.Tensor) -> dict:
        """omega_1 (B,) and harmonic-block frequencies (B, H)."""
        H = self.layout.H
        omega1 = self.freq_head(z, omega_hat)
        if self.frequency_mode == "locked":
            k = torch.arange(1, H + 1, device=z.device, dtype=z.dtype)
            harm = omega1[:, None] * k[None]
        elif self.frequency_mode == "free_ratio":
            kappa = (H + 0.5) * torch.sigmoid(self.kappa_logit)
            harm = omega1[:, None] * kappa[None]
        else:
            harm = self.free_head(z, omega_hat)
            # Lusch-style: heart rate is read from the first block's frequency.
            omega1 = harm[:, 0]
        return {"omega1": omega1, "harm": harm}

    # ------------------------------------------------------------------ propagation
    def propagate(self, z: torch.Tensor, steps, freqs: dict) -> torch.Tensor:
        return self.operator.propagate(z, steps, freqs["harm"])

    def step(self, z: torch.Tensor, freqs: dict) -> torch.Tensor:
        return self.operator.step(z, freqs["harm"])

    def volume(self, z: torch.Tensor) -> torch.Tensor:
        return self.readout(z)

    # ------------------------------------------------------------------ clinical readout
    def volume_grid(self, z_ref: torch.Tensor, freqs: dict, n_grid: int | None = None):
        """On-cycle volume on a grid for reference states z_ref (B, N, r).

        Returns (V (B, N, G), theta (G,) or (B, 1, G) phases, time offsets in frames).
        analytic: phase grid theta_n = 2 pi n / G (no omega dependence, Section III-D).
        general : time grid over one period of block 1, V from the closed-form trace.
        trace   : V of the propagated trajectory K^s z_ref, s = 0..trace_frames-1.
        """
        G = n_grid or self.n_phase_grid
        dev, dt = z_ref.device, z_ref.dtype
        if self.readout_mode == "analytic":
            c0, alpha, phi = self.readout.coefficients(z_ref)
            theta = fn.phase_grid(G, dev, dt)
            return fn.eval_trig_phase(c0, alpha, phi, theta), theta, None
        if self.readout_mode == "general":
            c0, alpha, phi = self.readout.coefficients(z_ref)
            w_ref = freqs["harm"][:, 0]  # (B,)
            frac = torch.arange(G, device=dev, dtype=dt) / G
            t = frac[None, :] * (2 * math.pi / w_ref[:, None])  # (B, G)
            om = freqs["harm"][:, None, :, None]  # (B, 1, H, 1)
            arg = om * t[:, None, None, :] + phi[..., None]
            V = c0[..., None] + (alpha[..., None] * torch.cos(arg)).sum(-2)
            return V, (2 * math.pi * frac)[None, None, :].expand(z_ref.shape[0], 1, G), t
        # trace mode
        S = self.trace_frames
        traj = self.operator.propagate(z_ref, list(range(S)), freqs["harm"])  # (B, N, S, r)
        V = self.readout(traj)
        t = torch.arange(S, device=dev, dtype=dt)
        theta = torch.remainder(freqs["omega1"][:, None, None] * t[None, None, :], 2 * math.pi)
        return V, theta, t

    def clinical_soft(self, z_ref: torch.Tensor, freqs: dict, beta: float) -> dict:
        """Differentiable EDV/ESV/EF and soft phases for z_ref (B, N, r)."""
        V, theta, _ = self.volume_grid(z_ref, freqs)
        out = fn.soft_extrema(V, beta, theta if theta.ndim == 1 else None)
        if theta.ndim > 1:
            for name, sgn in (("theta_ed", 1.0), ("theta_es", -1.0)):
                p = torch.softmax(sgn * beta * V, -1)
                out[name] = torch.remainder(
                    torch.atan2((p * torch.sin(theta)).sum(-1), (p * torch.cos(theta)).sum(-1)), 2 * math.pi)
        out["ef"] = fn.ejection_fraction(out["edv"], out["esv"])
        out["V"] = V
        return out

    @torch.no_grad()
    def clinical_exact(self, z_ref: torch.Tensor, freqs: dict, dense_grid: int = 256,
                       method: str = "roots", beta: float = 40.0) -> dict:
        """Inference readout for z_ref (B, r).

        method: 'roots' (exact degree-2H root finding; analytic mode only),
                'grid'  (hard extremum on a dense grid of ``dense_grid`` points),
                'soft'  (soft extremum with ``beta``, as in training).
        Non-analytic variants always fall back to the dense-grid hard extremum.
        """
        z3 = z_ref[:, None, :]
        if method == "roots" and self.readout_mode == "analytic":
            c0, alpha, phi = self.readout.coefficients(z_ref)
            ex = fn.exact_extrema(c0, alpha, phi)
            edv, esv = ex["edv"].to(z_ref), ex["esv"].to(z_ref)
            th_ed, th_es = ex["theta_ed"].to(z_ref), ex["theta_es"].to(z_ref)
        elif method == "soft":
            o = self.clinical_soft(z3, freqs, beta)
            edv, esv, th_ed, th_es = (o[k][:, 0] for k in ("edv", "esv", "theta_ed", "theta_es"))
        else:
            V, theta, _ = self.volume_grid(z3, freqs, dense_grid)
            V = V[:, 0]
            imax, imin = V.argmax(-1), V.argmin(-1)
            edv = V.gather(-1, imax[:, None])[:, 0]
            esv = V.gather(-1, imin[:, None])[:, 0]
            th = theta if theta.ndim == 1 else theta.reshape(theta.shape[0], -1)
            if th.ndim == 1:
                th_ed, th_es = th[imax], th[imin]
            else:
                th_ed = th.gather(-1, imax[:, None])[:, 0]
                th_es = th.gather(-1, imin[:, None])[:, 0]
        ef = fn.ejection_fraction(edv, esv)
        return {"edv": edv, "esv": esv, "ef": ef, "theta_ed": th_ed, "theta_es": th_es}
