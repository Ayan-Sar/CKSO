"""Test-time perturbations for the robustness study (Appendix E, last table).

Each factory returns a callable (frames (T, H, W) float, fps) -> (frames, fps), applied to
raw frames before resizing / normalisation. The exact perturbation definitions are not
given in the manuscript; ours are documented here and in docs/IMPLEMENTATION_NOTES.md.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from .frequency import detect_omega


def temporal_downsample(factor: int = 2):
    """Keep every ``factor``-th frame (e.g. 50 -> 25 fps)."""
    def f(frames, fps):
        return frames[::factor], fps / factor
    return f


def speckle(snr_db: float = 8.0, seed: int = 0):
    """Multiplicative Gaussian noise x * (1 + n), n ~ N(0, s^2), with signal-to-noise
    power ratio E[x^2] / E[(x n)^2] = 1 / s^2 = 10^{snr_db / 10}."""
    s = 10 ** (-snr_db / 20.0)
    def f(frames, fps):
        rng = np.random.RandomState(seed)
        out = frames * (1.0 + s * rng.randn(*frames.shape).astype(np.float32))
        return np.clip(out, 0, None), fps
    return f


def blur(sigma: float = 1.5):
    def f(frames, fps):
        return gaussian_filter(frames, sigma=(0, sigma, sigma)), fps
    return f


def sector_occlusion(fraction: float = 0.15, half_angle_deg: float = 45.0):
    """Zero a wedge covering ``fraction`` of the imaging sector (apex at top centre,
    sector spanning +-half_angle_deg), taken from the left edge of the sector."""
    def f(frames, fps):
        T, H, W = frames.shape
        yy, xx = np.mgrid[0:H, 0:W]
        ang = np.degrees(np.arctan2(xx - W / 2.0, yy + 1e-6))  # 0 = straight down
        lo = -half_angle_deg
        hi = lo + 2 * half_angle_deg * fraction
        mask = (ang >= lo) & (ang < hi)
        out = frames.copy()
        out[:, mask] = 0
        return out, fps
    return f


def truncate_cycles(n_cycles: int = 1, min_frames: int = 8):
    """Keep only the first ``n_cycles`` detected cycles (unsupervised detector)."""
    def f(frames, fps):
        est = detect_omega(frames, fps)
        keep = max(min_frames, int(math.ceil(n_cycles * est.period)) + 4)
        return frames[:keep], fps
    return f


PERTURBATIONS = {
    "none": lambda **k: None,
    "downsample2": lambda **k: temporal_downsample(2),
    "speckle8db": lambda **k: speckle(8.0, k.get("seed", 0)),
    "blur1.5": lambda **k: blur(1.5),
    "occlusion15": lambda **k: sector_occlusion(0.15),
    "truncate1": lambda **k: truncate_cycles(1),
}
