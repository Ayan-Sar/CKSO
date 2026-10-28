"""Self-contained walkthrough of the CKSO building blocks on one synthetic clip (CPU, seconds).

    python examples/synthetic_demo.py

1. render a synthetic cine loop with known HR / EF / ED / ES;
2. run the unsupervised autocorrelation detector (and the two Table IV baselines);
3. build the harmonic-lock operator and verify ||K^s||_2 = 1 and closed-form powers;
4. read EF and ED/ES phases analytically from a trigonometric volume trace.
No training happens here; the numbers printed are properties of this one synthetic clip.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ckso import functionals as fn  # noqa: E402
from ckso.data.synthetic import SyntheticSpec, event_frames, render  # noqa: E402
from ckso.frequency import detect_omega, omega_to_bpm  # noqa: E402
from ckso.layout import LatentLayout  # noqa: E402
from ckso.operator import CKSOOperator, power_norms  # noqa: E402


def main():
    spec = SyntheticSpec(hr_bpm=78.0, fps=50.0, n_frames=150, edv=130.0, esv=55.0, r2=0.25, p2=0.7, phi0=0.4)
    frames = render(spec) / 255.0
    ed, es = event_frames(spec)
    print(f"ground truth: HR {spec.hr_bpm:.1f} bpm, EF {spec.ef:.2f} %, ED frames {ed}, ES frames {es}")

    for meth in ("autocorr", "periodogram", "motion_autocorr"):
        est = detect_omega(frames, spec.fps, H=5, method=meth)
        print(f"detector {meth:16s}: period {est.period:6.2f} frames -> HR {omega_to_bpm(est.omega, spec.fps):6.1f} bpm")

    lay = LatentLayout(5, 4, 3)
    op = CKSOOperator(lay).double()
    w1 = torch.tensor([spec.omega], dtype=torch.float64)
    K = op.matrix(w1[:, None] * torch.arange(1, 6, dtype=torch.float64))[0]
    norms = power_norms(K, 64)
    print(f"operator: r = {lay.r}, {lay.n_operator_scalars} scalars, max_s<=64 ||K^s||_2 = {float(norms.max()):.12f}")

    # The synthetic V(theta) is an affine map of cos(th) + r2 cos(2 th + p2): write its coefficients.
    th = np.linspace(0, 2 * np.pi, 20001)
    s = np.cos(th) + spec.r2 * np.cos(2 * th + spec.p2) + spec.r3 * np.cos(3 * th + spec.p3)
    scale = (spec.edv - spec.esv) / (s.max() - s.min())
    c0 = torch.tensor([spec.esv - scale * s.min()], dtype=torch.float64)
    alpha = torch.tensor([[scale, scale * spec.r2, scale * spec.r3]], dtype=torch.float64)
    phi = torch.tensor([[spec.phi0, 2 * spec.phi0 + spec.p2, 3 * spec.phi0 + spec.p3]], dtype=torch.float64)
    ex = fn.exact_extrema(c0, alpha, phi)
    ef = float(fn.ejection_fraction(ex["edv"], ex["esv"])[0])
    t_ed = float(ex["theta_ed"][0]) / spec.omega
    t_es = float(ex["theta_es"][0]) / spec.omega
    print(f"analytic readout: EDV {float(ex['edv'][0]):.2f}, ESV {float(ex['esv'][0]):.2f}, EF {ef:.2f} %, "
          f"first ED at t = {t_ed:.2f}, first ES at t = {t_es:.2f} frames (period {2 * math.pi / spec.omega:.2f})")
    gp = fn.phase_grid(64, dtype=torch.float64)
    soft = fn.soft_extrema(fn.eval_trig_phase(c0, alpha, phi, gp), 40.0, gp)
    bound = float(fn.soft_extremum_bound(64, 40.0, alpha)[0])
    print(f"soft extremum (beta=40, N=64): EDV {float(soft['edv'][0]):.2f}; |error| "
          f"{abs(float(soft['edv'][0]) - float(ex['edv'][0])):.3f} <= bound {bound:.3f}")


if __name__ == "__main__":
    main()
