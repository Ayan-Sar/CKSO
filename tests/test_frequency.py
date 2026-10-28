import math

import numpy as np

from ckso.data.synthetic import SyntheticSpec, render
from ckso.frequency import (autocorrelation, detect_omega, detect_period_autocorr, detect_period_periodogram,
                            motion_energy_surrogate, mean_intensity_surrogate, omega_to_bpm)


def harmonic_signal(period, n=300, r2=0.8, noise=0.0, seed=0):
    t = np.arange(n)
    w = 2 * np.pi / period
    rng = np.random.RandomState(seed)
    return np.cos(w * t) + r2 * np.cos(2 * w * t + 0.3) + noise * rng.randn(n)


def test_autocorrelation_normalised():
    acf = autocorrelation(np.random.randn(100))
    assert abs(acf[0] - 1) < 1e-12 and np.all(np.abs(acf) <= 1 + 1e-12)


def test_autocorr_detector_recovers_integer_period():
    for period in (14, 25, 40, 60):
        est = detect_period_autocorr(harmonic_signal(period), fps=50.0)
        assert est is not None and abs(est.period - period) <= 1


def test_autocorr_avoids_octave_error_where_periodogram_fails():
    # strong second harmonic: the periodogram selects 2*omega_1, the autocorrelation does not
    sig = harmonic_signal(40, r2=1.6)
    ac = detect_period_autocorr(sig, fps=50.0)
    pg = detect_period_periodogram(sig, fps=50.0)
    assert abs(ac.period - 40) <= 1
    assert abs(pg.period - 20) <= 1


def test_motion_energy_peaks_twice_per_cycle():
    spec = SyntheticSpec(hr_bpm=60.0, fps=50.0, n_frames=200, r2=0.0, r3=0.0, speckle=False, noise_std=0.0)
    frames = render(spec)
    me = motion_energy_surrogate(frames)
    est_me = detect_period_autocorr(me, 50.0, rel_height=0.5)
    est_mi = detect_period_autocorr(mean_intensity_surrogate(frames), 50.0)
    assert abs(est_mi.period - 50) <= 1
    assert abs(est_me.period - 25) <= 2  # octave error of the motion-energy surrogate


def test_detect_omega_on_synthetic_clip_and_clamp():
    spec = SyntheticSpec(hr_bpm=90.0, fps=50.0, n_frames=160)
    est = detect_omega(render(spec), 50.0, H=5)
    assert est.ok
    assert abs(omega_to_bpm(est.omega, 50.0) - 90.0) < 6.0  # integer-lag quantisation
    # clamp below omega_max for very high rates
    spec2 = SyntheticSpec(hr_bpm=200.0, fps=25.0, n_frames=100)
    est2 = detect_omega(render(spec2), 25.0, H=5)
    assert est2.omega < math.pi / 5.5


def test_fallback_flagged():
    est = detect_omega(np.zeros((40, 16, 16), np.float32), 50.0)
    assert not est.ok and est.method == "fallback"
