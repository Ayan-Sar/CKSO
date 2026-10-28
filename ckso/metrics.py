"""Evaluation metrics (Section IV)."""

from __future__ import annotations

import numpy as np


def _finite(*arrs):
    arrs = [np.asarray(a, dtype=np.float64) for a in arrs]
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return [a[m] for a in arrs]


def mae(y, p):
    y, p = _finite(y, p)
    return float(np.mean(np.abs(p - y))) if len(y) else float("nan")


def rmse(y, p):
    y, p = _finite(y, p)
    return float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else float("nan")


def r2(y, p):
    y, p = _finite(y, p)
    if len(y) < 2:
        return float("nan")
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pearson(y, p):
    y, p = _finite(y, p)
    if len(y) < 2 or y.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def frac_abs_error_above(y, p, thr=15.0):
    y, p = _finite(y, p)
    return float(np.mean(np.abs(p - y) > thr)) if len(y) else float("nan")


def ef_metrics(y, p) -> dict:
    y, p = _finite(y, p)
    return {"n": int(len(y)), "mae": mae(y, p), "rmse": rmse(y, p), "r2": r2(y, p),
            "bias": float(np.mean(p - y)) if len(y) else float("nan"),
            "frac_abs_err_gt15": frac_abs_error_above(y, p, 15.0)}


def octave_error_rate(hr_true, hr_pred, tol: float = 0.15) -> float:
    """Fraction with |log2(pred/true)| within ``tol`` of 1 (factor-2 errors either way)."""
    t, p = _finite(hr_true, hr_pred)
    ok = (t > 0) & (p > 0)
    t, p = t[ok], p[ok]
    if not len(t):
        return float("nan")
    lr = np.abs(np.log2(p / t))
    return float(np.mean(np.abs(lr - 1.0) < tol))


def hr_metrics(hr_true, hr_pred) -> dict:
    t, p = _finite(hr_true, hr_pred)
    return {"n": int(len(t)), "hr_mae_bpm": mae(t, p), "octave_error_rate": octave_error_rate(t, p),
            "pearson_r": pearson(t, p)}


def stratified_mae(y, p, bands=((-np.inf, 30), (30, 40), (40, 55), (55, 70), (70, np.inf))) -> dict:
    y, p = _finite(y, p)
    out = {}
    for lo, hi in bands:
        m = (y >= lo) & (y < hi)
        key = f"[{lo},{hi})"
        out[key] = {"n": int(m.sum()), "mae": mae(y[m], p[m]) if m.any() else float("nan")}
    return out


def afd_summary(afd_values) -> dict:
    a = np.asarray([v for v in afd_values if np.isfinite(v)], dtype=np.float64)
    if not len(a):
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {"n": int(len(a)), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def binary_classification(y_true, score, threshold: float | None = None) -> dict:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=np.float64)
    out = {"n": int(len(y))}
    out["auc"] = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
    if threshold is not None:
        pr = (s >= threshold).astype(int)
        tp = int(((pr == 1) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum())
        fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
        out.update(accuracy=(tp + tn) / max(len(y), 1), sensitivity=tp / max(tp + fn, 1),
                   specificity=tn / max(tn + fp, 1))
    return out
