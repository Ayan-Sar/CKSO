"""Per-clip inference and dataset-level evaluation (Section IV, Tables I, III, IV; Section V).

Per clip:
  * encode every delay window, refine omega_1 with the head;
  * split the clip into beats of the learned period T_c = 2 pi / omega_1; for every beat
    start j_b take the encoded state z_{j_b} and read EDV/ESV/EF/ED/ES from its on-cycle
    trigonometric polynomial (exact root finding by default);
  * protocol 'central' / 'three' / 'all' averages EF over 1, 3 or all beats (Table III);
  * ED/ES events: j_b + (m - 1) + theta / omega_1 for each beat (z_j encodes the window
    ending at frame j + m - 1); aFD = distance from each labelled frame to the nearest
    predicted event (no phase labels are used for prediction);
  * diagnostics: linearity residual L_lin, frequency disagreement |omega_1 - omega_hat| /
    omega_hat, per-block empirical moduli (spectral descriptor), detector baselines.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from .frequency import detect_omega, omega_to_bpm
from . import metrics as M


def _beat_starts(n_states: int, period: float) -> list[int]:
    starts, b = [], 0
    while True:
        j = int(round(b * period))
        if j + period > n_states + 1e-6:
            break
        starts.append(j)
        b += 1
    return starts or [0]


def _select(starts: list[int], n_states: int, period: float, protocol: str) -> list[int]:
    if protocol == "all" or len(starts) == 1:
        return starts
    centers = np.array([j + period / 2 for j in starts])
    order = np.argsort(np.abs(centers - n_states / 2))
    k = 1 if protocol == "central" else 3
    return sorted(starts[i] for i in order[:k])


def empirical_block_moduli(z: torch.Tensor, layout) -> np.ndarray:
    """Per-block least-squares one-step eigenvalue modulus |lambda_k| of the encoded
    trajectory (a per-clip, DMD-style descriptor; the learned rho_k are global)."""
    _, harm, _, _ = layout.split(z)
    c = torch.complex(harm[..., 0].double(), harm[..., 1].double())  # (T', H)
    num = (c[:-1].conj() * c[1:]).sum(0)
    den = (c[:-1].abs() ** 2).sum(0) + 1e-12
    return (num / den).abs().cpu().numpy()


@torch.no_grad()
def predict_clip(model, frames: torch.Tensor, omega_hat: float, fps: float, protocol: str = "all",
                 readout: str = "roots", beta: float = 40.0, chunk: int = 256,
                 ed_frames=(), es_frames=(), baselines: bool = True) -> dict:
    dev = next(model.parameters()).device
    x = frames[None].to(dev).float()
    z = model.encode(x, chunk)  # (1, T', r)
    wh = torch.tensor([float(omega_hat)], device=dev)
    freqs = model.frequencies(z, wh)
    w1 = float(freqs["omega1"][0])
    period = 2 * math.pi / w1
    Tp = z.shape[1]
    m = model.m

    Kz = model.step(z[:, :-1], freqs)
    lin = float(((z[:, 1:] - Kz) ** 2).sum(-1).mean())
    lin_norm = lin / float((z ** 2).sum(-1).mean() + 1e-12)

    starts = _select(_beat_starts(Tp, period), Tp, period, protocol)
    js = torch.tensor(starts, device=dev)
    nb = len(starts)
    fb = {k: v.expand(nb, *v.shape[1:]) for k, v in freqs.items()}
    cl = model.clinical_exact(z[0, js], fb, method=readout, beta=beta)
    ef_beats = cl["ef"].cpu().numpy()

    # Phase events over the whole clip (all beats, regardless of the EF protocol).
    all_starts = _beat_starts(Tp, period)
    ja = torch.tensor(all_starts, device=dev)
    fa = {k: v.expand(len(all_starts), *v.shape[1:]) for k, v in freqs.items()}
    ca = model.clinical_exact(z[0, ja], fa, method=readout, beta=beta)
    ev = {}
    T = frames.shape[0]
    for name in ("ed", "es"):
        off = (ca[f"theta_{name}"].cpu().numpy() % (2 * math.pi)) / w1
        base = np.array(all_starts) + (m - 1) + off
        # Extrapolate one period backwards (frames before the first full delay window)
        # and forwards to the end of the clip (partial last beat).
        pre = base[0] - period * np.arange(1, 3)
        n_post = int(math.ceil(max(0.0, T - base[-1]) / period)) + 1
        post = base[-1] + period * np.arange(1, n_post + 1)
        evs = np.sort(np.concatenate([pre, base, post]))
        ev[name] = evs[(evs > -period) & (evs < T + period)]

    def afd(labels, events):
        return [float(np.min(np.abs(events - f))) for f in labels] if len(events) else [float("nan")] * len(labels)

    if model.readout_mode == "analytic":
        _, alpha, _ = model.readout.coefficients(z[0, ja])
        alpha_mean = alpha.mean(0).cpu().numpy()
    else:
        alpha_mean = np.full(model.layout.H, np.nan)
    rho_emp = empirical_block_moduli(z[0], model.layout)

    out = {
        "ef_pred": float(ef_beats.mean()),
        "ef_pred_beats": ef_beats.tolist(),
        "edv_pred": float(cl["edv"].mean()),
        "esv_pred": float(cl["esv"].mean()),
        "n_beats_used": nb,
        "omega1": w1,
        "omega_hat": float(omega_hat),
        "hr_pred_bpm": float(omega_to_bpm(w1, fps)),
        "hr_detector_bpm": float(omega_to_bpm(float(omega_hat), fps)),
        "freq_disagreement": abs(w1 - float(omega_hat)) / float(omega_hat),
        "lin_residual": lin,
        "lin_residual_norm": lin_norm,
        "period_frames": period,
        "ed_events": ev["ed"].tolist(),
        "es_events": ev["es"].tolist(),
        "ed_afd": afd(list(ed_frames), ev["ed"]),
        "es_afd": afd(list(es_frames), ev["es"]),
        "alpha": alpha_mean.tolist(),
        "rho_emp": rho_emp.tolist(),
        "volume_trace": model.volume(z[0]).cpu().numpy().tolist(),
    }
    if baselines:
        fr = frames.cpu().numpy()
        for meth in ("periodogram", "motion_autocorr"):
            e = detect_omega(fr, fps, H=None, method=meth)
            out[f"hr_{meth}_bpm"] = float(omega_to_bpm(e.omega, fps))
    return out


def evaluate_dataset(model, dataset, protocol: str = "all", readout: str = "roots", beta: float = 40.0,
                     max_clips: int | None = None, progress: bool = True) -> tuple[pd.DataFrame, dict]:
    rows = []
    it = range(len(dataset) if max_clips is None else min(max_clips, len(dataset)))
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc="eval")
        except ImportError:  # pragma: no cover
            pass
    for i in it:
        item = dataset[i]
        res = predict_clip(model, item["frames"], float(item["omega_hat"]), float(item["fps"]), protocol, readout,
                           beta, ed_frames=item["ed_frames"], es_frames=item["es_frames"])
        res.update(clip_id=item["clip_id"], patient_id=item["patient_id"], view=item["view"],
                   ef_true=float(item["ef"]), edv_true=float(item["edv"]), esv_true=float(item["esv"]),
                   hr_true_bpm=float(item["hr_bpm"]), label=float(item["label"]),
                   detector_ok=bool(item["detector_ok"]), fps=float(item["fps"]),
                   quality=str(dataset.records[i].extra.get("quality", "")))
        rows.append(res)
    df = pd.DataFrame(rows)
    return df, summarize(df)


def summarize(df: pd.DataFrame) -> dict:
    s = {"n_clips": int(len(df))}
    if not len(df):
        return s
    if np.isfinite(df["ef_true"]).any():
        s["ef"] = M.ef_metrics(df["ef_true"], df["ef_pred"])
        s["ef_stratified"] = M.stratified_mae(df["ef_true"], df["ef_pred"])
    if np.isfinite(df["hr_true_bpm"]).any():
        s["hr"] = {name: M.hr_metrics(df["hr_true_bpm"], df[col]) for name, col in
                   (("learned_omega1", "hr_pred_bpm"), ("autocorr_detector", "hr_detector_bpm"),
                    ("periodogram", "hr_periodogram_bpm"), ("motion_autocorr", "hr_motion_autocorr_bpm"))
                   if col in df}
    else:
        s["hr"] = "unavailable: no reference heart rate for this dataset (see README)"
    ed = [v for lst in df["ed_afd"] for v in lst]
    es = [v for lst in df["es_afd"] for v in lst]
    s["ed_afd"] = M.afd_summary(ed)
    s["es_afd"] = M.afd_summary(es)
    s["lin_residual_median"] = float(np.median(df["lin_residual"]))
    s["freq_disagreement_median"] = float(np.median(df["freq_disagreement"]))
    s["detector_fallback_rate"] = float(1 - df["detector_ok"].mean())
    return s
