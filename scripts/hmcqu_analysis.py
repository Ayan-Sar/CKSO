"""HMC-QU analyses (Section IV-F, Nyquist-binding regime).

trace : for clips with per-frame masks, correlate the recovered V(t) (encoded states,
        frames m-1 .. T-1) with the mask-derived cavity area: Pearson r, normalised RMSE
        after a least-squares affine fit (area units are arbitrary), fraction of the
        reference trace energy in harmonics k <= H, and ED/ES localisation against the
        area maxima/minima of the reference trace.
mi    : linear SVM on the 10-number spectral descriptor
        (rho_emp_1..3, alpha_1..3, alpha_2/alpha_1, alpha_3/alpha_1, analytic EF, L_lin)
        with stratified, patient-level K-fold CV. rho are per-clip empirical moduli
        (the learned rho_k are global parameters, identical for every clip).

    python scripts/hmcqu_analysis.py trace --ckpt runs/hmcqu/fold0/best.pt --split test --out runs/hmcqu/trace_fold0.csv
    python scripts/hmcqu_analysis.py mi --pred runs/hmcqu/fold*/eval_test/predictions.csv --out runs/hmcqu/mi.json
"""

import argparse
import glob
import math

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from _common import add_common_args, load_predictions
from ckso.config import apply_overrides
from ckso.data import build_dataset
from ckso.evaluation import predict_clip
from ckso.metrics import binary_classification
from ckso.training import load_model
from ckso.utils import get_device, save_json


def harmonic_energy_fraction(x: np.ndarray, period: float, H: int) -> float:
    """Fraction of (mean-removed) energy at harmonics k <= H of the fundamental 1/period,
    by least-squares fit of a degree-H trigonometric polynomial."""
    t = np.arange(len(x))
    cols = [np.ones_like(t, dtype=float)]
    for k in range(1, H + 1):
        w = 2 * np.pi * k / period
        cols += [np.cos(w * t), np.sin(w * t)]
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    fit = A @ coef
    xc = x - x.mean()
    return float(np.sum((fit - x.mean()) ** 2) / max(np.sum(xc ** 2), 1e-12))


def run_trace(a):
    dev = get_device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    cfg = apply_overrides(cfg, a.override)
    ds = build_dataset(cfg, a.split, "eval")
    rows = []
    for i, rec in enumerate(ds.records):
        area = ds.cavity_area_trace(rec)
        if area is None:
            continue
        item = ds[i]
        T = item["frames"].shape[0]
        area = area[:T]
        # reference ED/ES = local maxima / minima of the reference area trace
        per = 2 * math.pi / float(item["omega_hat"])
        ed_ref = find_peaks(area, distance=max(1, int(0.6 * per)))[0].tolist()
        es_ref = find_peaks(-area, distance=max(1, int(0.6 * per)))[0].tolist()
        res = predict_clip(model, item["frames"], float(item["omega_hat"]), float(item["fps"]),
                           ed_frames=ed_ref, es_frames=es_ref, baselines=False)
        v = np.asarray(res["volume_trace"])
        ref = area[model.m - 1: model.m - 1 + len(v)]
        n = min(len(v), len(ref))
        v, ref = v[:n], ref[:n]
        r = float(np.corrcoef(v, ref)[0, 1]) if n > 2 and v.std() > 0 and ref.std() > 0 else float("nan")
        A = np.stack([v, np.ones_like(v)], 1)
        coef, *_ = np.linalg.lstsq(A, ref, rcond=None)
        nrmse = float(np.sqrt(np.mean((A @ coef - ref) ** 2)) / max(ref.max() - ref.min(), 1e-12))
        rows.append({"clip_id": rec.clip_id, "pearson_r": r, "nrmse": nrmse,
                     "ref_energy_k_le_H": harmonic_energy_fraction(area.astype(float), per, model.layout.H),
                     "ed_afd_mean": float(np.nanmean(res["ed_afd"])) if res["ed_afd"] else float("nan"),
                     "es_afd_mean": float(np.nanmean(res["es_afd"])) if res["es_afd"] else float("nan")})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    if len(df):
        print(df.describe().loc[["count", "mean", "50%", "std"]])
    else:
        print("No clips with masks found (set data.kwargs.mask_dir).")


def descriptor(df: pd.DataFrame) -> np.ndarray:
    al = np.array([x[:3] + [np.nan] * (3 - len(x[:3])) for x in df["alpha"]], float)
    rh = np.array([x[:3] + [np.nan] * (3 - len(x[:3])) for x in df["rho_emp"]], float)
    a1 = np.where(np.abs(al[:, 0]) > 1e-12, al[:, 0], np.nan)
    return np.column_stack([rh, al, al[:, 1] / a1, al[:, 2] / a1, df["ef_pred"], df["lin_residual"]])


def run_mi(a):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    files = sorted({f for pat in a.pred for f in glob.glob(pat)})
    df = pd.concat([load_predictions(f) for f in files]).drop_duplicates("clip_id").reset_index(drop=True)
    df = df[np.isfinite(df["label"])].reset_index(drop=True)
    X = descriptor(df)
    ok = np.all(np.isfinite(X), 1)
    X, y, g = X[ok], df["label"].to_numpy()[ok].astype(int), df["patient_id"].astype(str).to_numpy()[ok]
    subsets = {"full_descriptor": slice(None), "harmonic_ratios": [6, 7], "ef_only": [8]}
    res = {"n": int(len(y)), "n_mi": int(y.sum()), "files": files}
    for name, cols in subsets.items():
        accs, aucs, scores = [], [], np.zeros(len(y))
        for rep in range(a.repeats):
            cv = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=rep)
            fold_acc = []
            for tr, te in cv.split(X, y, g):
                clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=a.C))
                clf.fit(X[tr][:, cols], y[tr])
                scores[te] = clf.decision_function(X[te][:, cols])
                fold_acc.append(float((clf.predict(X[te][:, cols]) == y[te]).mean()))
            m = binary_classification(y, scores, threshold=0.0)
            accs.append(m["accuracy"])
            aucs.append(m["auc"])
            if rep == 0:
                res[name] = {"rep0": m, "fold_accuracy": fold_acc}
        res[name]["accuracy_mean"] = float(np.mean(accs))
        res[name]["accuracy_std"] = float(np.std(accs))
        res[name]["auc_mean"] = float(np.mean(aucs))
        res[name]["auc_std"] = float(np.std(aucs))
    save_json(res, a.out)
    print({k: (v["auc_mean"], v["accuracy_mean"]) for k, v in res.items() if isinstance(v, dict) and "auc_mean" in v})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = add_common_args(sub.add_parser("trace"))
    t.add_argument("--ckpt", required=True)
    t.add_argument("--split", default="test")
    t.add_argument("--out", required=True)
    m = sub.add_parser("mi")
    m.add_argument("--pred", nargs="+", required=True)
    m.add_argument("--folds", type=int, default=5)
    m.add_argument("--repeats", type=int, default=10)
    m.add_argument("--C", type=float, default=1.0)
    m.add_argument("--out", required=True)
    a = p.parse_args()
    run_trace(a) if a.cmd == "trace" else run_mi(a)


if __name__ == "__main__":
    main()
