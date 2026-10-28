"""Subject-conditioned fundamental frequency (Section III-C).

Detector (unsupervised, numpy):
  s_t = mean intensity of a central region (one cycle per beat; the cavity is
  hypoechoic, contraction raises s_t). The period is the smallest *prominent* lag of the
  autocorrelation inside the physiological band [f_s/f_max, f_s/f_min], omega_hat =
  2 pi / tau*. The lag is an integer (the quantisation the refinement head corrects).

Baselines for Table IV are also provided: periodogram peak of the same surrogate, and
the motion-energy surrogate sum|x_t - x_{t-1}| (which peaks twice per cycle).

Refinement head (torch):
  u = u0(omega_hat) + MLP(z_bar, delta_bar, omega_hat), last layer zero-initialised,
  u0 = log(omega_hat / (omega_max - omega_hat)), omega_1 = omega_max * sigmoid(u),
  so omega_1 == omega_hat at step 0 and H*omega_1 < pi for every u.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from scipy.signal import detrend, find_peaks

from .operator import omega_max as _omega_max


# --------------------------------------------------------------------------- surrogates
def mean_intensity_surrogate(frames: np.ndarray, center_frac: float = 0.5) -> np.ndarray:
    """Mean intensity of the central (center_frac x center_frac) region. frames (T, H, W)."""
    T, Hh, Ww = frames.shape
    h0 = int(round(Hh * (1 - center_frac) / 2))
    w0 = int(round(Ww * (1 - center_frac) / 2))
    roi = frames[:, h0:Hh - h0, w0:Ww - w0]
    return roi.reshape(T, -1).mean(1).astype(np.float64)


def motion_energy_surrogate(frames: np.ndarray) -> np.ndarray:
    """sum |x_t - x_{t-1}| (length T-1). Peaks twice per cycle -> octave errors."""
    d = np.abs(np.diff(frames.astype(np.float64), axis=0))
    return d.reshape(d.shape[0], -1).sum(1)


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised (biased) autocorrelation via FFT; acf[0] = 1."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)[:n]
    if acf[0] <= 0:
        return np.zeros(n)
    return acf / acf[0]


@dataclass
class PeriodEstimate:
    period: float          # frames
    omega: float           # rad / frame
    method: str
    ok: bool               # False if a fallback was used


def lag_band(fps: float, f_min: float = 0.5, f_max: float = 3.7) -> tuple[int, int]:
    return int(math.floor(fps / f_max)), int(math.ceil(fps / f_min))


def detect_period_autocorr(signal: np.ndarray, fps: float, f_min: float = 0.5, f_max: float = 3.7,
                           rel_height: float = 0.8, min_prominence: float = 0.05,
                           max_lag_frac: float = 0.67) -> PeriodEstimate | None:
    """Smallest prominent autocorrelation lag in the physiological band.

    A peak is "prominent" if its scipy prominence >= min_prominence and its height is at
    least rel_height x the highest in-band peak. The smallest such lag is returned, which
    rejects sub-harmonic peaks at multiples of the period. Returns None if no peak exists.
    """
    x = detrend(np.asarray(signal, dtype=np.float64))
    acf = autocorrelation(x)
    lo, hi = lag_band(fps, f_min, f_max)
    hi = min(hi, int(max_lag_frac * len(x)))
    lo = max(lo, 2)
    if hi <= lo + 1:
        return None
    peaks, props = find_peaks(acf[: hi + 1], prominence=min_prominence)
    peaks = [p for p in peaks if lo <= p <= hi and acf[p] > 0]
    if not peaks:
        return None
    heights = np.array([acf[p] for p in peaks])
    keep = [p for p, h in zip(peaks, heights) if h >= rel_height * heights.max()]
    tau = float(min(keep))
    return PeriodEstimate(tau, 2 * math.pi / tau, "autocorr", True)


def detect_period_periodogram(signal: np.ndarray, fps: float, f_min: float = 0.5, f_max: float = 3.7,
                              pad_factor: int = 8) -> PeriodEstimate | None:
    """Periodogram peak in the physiological band (Table IV baseline; prone to 2*omega)."""
    x = detrend(np.asarray(signal, dtype=np.float64)) * np.hanning(len(signal))
    n = len(x) * pad_factor
    P = np.abs(np.fft.rfft(x, n)) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / fps)
    band = (f >= f_min) & (f <= f_max)
    if not band.any() or P[band].max() <= 0:
        return None
    fpk = f[band][np.argmax(P[band])]
    tau = fps / fpk
    return PeriodEstimate(tau, 2 * math.pi / tau, "periodogram", True)


def detect_omega(frames: np.ndarray, fps: float, H: int | None = None, method: str = "autocorr",
                 center_frac: float = 0.5, f_min: float = 0.5, f_max: float = 3.7,
                 rel_height: float = 0.8, min_prominence: float = 0.05,
                 fallback_bpm: float = 75.0, margin: float = 0.98) -> PeriodEstimate:
    """omega_hat (rad/frame) for a clip of frames (T, H, W) at ``fps``.

    method: 'autocorr' (CKSO initializer), 'periodogram' (Table IV row 1),
            'motion_autocorr' (Table IV row 2).
    If H is given, omega_hat is clamped to < margin * omega_max(H) so that the logit
    initialisation of the refinement head is finite.
    """
    if method == "motion_autocorr":
        sig = motion_energy_surrogate(frames)
    else:
        sig = mean_intensity_surrogate(frames, center_frac)
    est = None
    if len(sig) >= 8 and np.std(sig) > 0:
        if method in ("autocorr", "motion_autocorr"):
            est = detect_period_autocorr(sig, fps, f_min, f_max, rel_height, min_prominence)
            if est is not None:
                est.method = method
        elif method == "periodogram":
            est = detect_period_periodogram(sig, fps, f_min, f_max)
        else:
            raise ValueError(f"Unknown method {method}")
    if est is None:
        # Fallback, flagged ok=False: periodogram, then a nominal heart rate.
        est = detect_period_periodogram(sig, fps, f_min, f_max) if len(sig) >= 8 and np.std(sig) > 0 else None
        if est is None:
            tau = fps * 60.0 / fallback_bpm
            est = PeriodEstimate(tau, 2 * math.pi / tau, "fallback", False)
        else:
            est.ok = False
            est.method = "fallback_periodogram"
    if H is not None:
        wmax = _omega_max(H)
        if est.omega >= margin * wmax:
            est.omega = margin * wmax
            est.period = 2 * math.pi / est.omega
    return est


def omega_to_bpm(omega: torch.Tensor | float, fps: torch.Tensor | float):
    """Heart rate in bpm from omega (rad/frame) and frame rate."""
    return omega * fps * 60.0 / (2 * math.pi)


def bpm_to_omega(bpm, fps):
    return 2 * math.pi * bpm / (60.0 * fps)


# --------------------------------------------------------------------------- learned head
class FrequencyHead(nn.Module):
    """Two-layer MLP refinement of the detector: (z_bar, delta_bar, omega_hat) -> u."""

    def __init__(self, r: int, H: int, hidden: int = 64, n_out: int = 1):
        super().__init__()
        self.H = H
        self.omega_max = _omega_max(H)
        self.n_out = n_out
        self.fc1 = nn.Linear(2 * r + 1, hidden)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(hidden, n_out)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def summary(z: torch.Tensor):
        """z_bar = mean_t z_t, delta_bar = mean_t |z_{t+1} - z_t| for z (B, T, r).

        The manuscript does not define delta_bar precisely; the mean absolute latent
        increment is our reading (documented in docs/IMPLEMENTATION_NOTES.md)."""
        zbar = z.mean(1)
        dbar = (z[:, 1:] - z[:, :-1]).abs().mean(1) if z.shape[1] > 1 else torch.zeros_like(zbar)
        return zbar, dbar

    def u0(self, omega_hat: torch.Tensor) -> torch.Tensor:
        w = torch.clamp(omega_hat, 1e-6, self.omega_max * (1 - 1e-6))
        return torch.log(w) - torch.log(self.omega_max - w)

    def forward(self, z: torch.Tensor, omega_hat: torch.Tensor) -> torch.Tensor:
        zbar, dbar = self.summary(z)
        feat = torch.cat([zbar, dbar, omega_hat[:, None].to(z.dtype)], -1)
        delta = self.fc2(self.act(self.fc1(feat)))  # (B, n_out)
        u = self.u0(omega_hat)[:, None] + delta
        omega1 = self.omega_max * torch.sigmoid(u)
        return omega1[:, 0] if self.n_out == 1 else omega1


class FreeFrequencyHead(nn.Module):
    """Lusch-style ablation (Table V row 1): H free, subject-conditioned block frequencies
    omega_k = pi * sigmoid(u_k) in (0, pi), initialised at k * omega_hat."""

    def __init__(self, r: int, H: int, hidden: int = 64):
        super().__init__()
        self.H = H
        self.fc1 = nn.Linear(2 * r + 1, hidden)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(hidden, H)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z: torch.Tensor, omega_hat: torch.Tensor) -> torch.Tensor:
        zbar, dbar = FrequencyHead.summary(z)
        feat = torch.cat([zbar, dbar, omega_hat[:, None].to(z.dtype)], -1)
        k = torch.arange(1, self.H + 1, device=z.device, dtype=z.dtype)
        target = torch.clamp(k[None] * omega_hat[:, None] / math.pi, 1e-4, 1 - 1e-4)
        u0 = torch.log(target) - torch.log1p(-target)
        return math.pi * torch.sigmoid(u0 + self.fc2(self.act(self.fc1(feat))))
