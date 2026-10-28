"""Statistical analysis: paired bootstrap (Table II), Holm correction, ICC, seed aggregation."""

from __future__ import annotations

import numpy as np


def _mae(e):
    return np.mean(np.abs(e), axis=-1)


def _rmse(e):
    return np.sqrt(np.mean(e ** 2, axis=-1))


def paired_bootstrap(y, pred_a, pred_b, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired bootstrap of Delta = metric(a) - metric(b) for MAE and RMSE on the same cases.

    Negative Delta means method a has lower error. Two-sided p-value is
    2 * min(P(Delta* <= 0), P(Delta* >= 0)) over bootstrap replicates (percentile method).
    """
    y = np.asarray(y, float); a = np.asarray(pred_a, float); b = np.asarray(pred_b, float)
    m = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[m], a[m], b[m]
    n = len(y)
    if n < 2:
        raise ValueError("need >= 2 paired cases")
    ea, eb = a - y, b - y
    rng = np.random.RandomState(seed)
    out = {"n": n}
    for name, f in (("mae", _mae), ("rmse", _rmse)):
        point = float(f(ea) - f(eb))
        deltas = np.empty(n_boot)
        for i0 in range(0, n_boot, 1000):
            k = min(1000, n_boot - i0)
            idx = rng.randint(0, n, size=(k, n))
            deltas[i0:i0 + k] = f(ea[idx]) - f(eb[idx])
        lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        p = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
        out[name] = {"delta": point, "ci_low": float(lo), "ci_high": float(hi), "p": float(min(1.0, p))}
    return out


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def icc_2_1(X) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater.
    X: (n_subjects, k_raters) without missing values."""
    X = np.asarray(X, float)
    X = X[np.all(np.isfinite(X), 1)]
    n, k = X.shape
    if n < 2 or k < 2:
        return float("nan")
    gm = X.mean()
    msr = k * np.sum((X.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((X.mean(0) - gm) ** 2) / (k - 1)
    sse = np.sum((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den != 0 else float("nan")


def aggregate_seeds(results: list[dict]) -> dict:
    """mean +- s.d. (ddof=1) of every numeric leaf across seed result dicts."""
    out = {}
    keys = set().union(*[r.keys() for r in results]) if results else set()
    for k in sorted(keys):
        vals = [r.get(k) for r in results]
        if all(isinstance(v, dict) for v in vals if v is not None):
            out[k] = aggregate_seeds([v for v in vals if v is not None])
        else:
            num = np.array([v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)], float)
            if len(num):
                out[k] = {"mean": float(num.mean()), "std": float(num.std(ddof=1)) if len(num) > 1 else 0.0,
                          "n_seeds": int(len(num))}
    return out
