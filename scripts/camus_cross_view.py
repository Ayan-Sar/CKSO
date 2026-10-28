"""CAMUS cross-view consistency (Section IV-F): the same heart in A2C and A4C must give the
same omega_1. Reports mean |Delta HR| and ICC(2,1) between views for the learned omega_1
and the autocorrelation detector, plus EF MAE by image quality.

Caveat: CAMUS sequences are half cycles; with ``mirror_half_cycle`` the "period" is twice
the ED->ES duration of the mirrored clip, not a measured cardiac period. Interpret with care.

    python scripts/camus_cross_view.py --pred runs/camus/eval_test/predictions.csv --out runs/camus/cross_view.json
"""

import argparse

from _common import load_predictions
from ckso.metrics import ef_metrics
from ckso.stats import icc_2_1
from ckso.utils import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True)
    p.add_argument("--quality-csv", default=None, help="optional clip_id,quality if not in predictions")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    df = load_predictions(a.pred)
    res = {}
    piv = {}
    for col, name in (("hr_pred_bpm", "learned_omega1"), ("hr_detector_bpm", "autocorr_detector")):
        w = df.pivot_table(index="patient_id", columns="view", values=col)
        w = w.dropna()
        if w.shape[1] < 2:
            continue
        piv[name] = w
        d = (w.iloc[:, 0] - w.iloc[:, 1]).abs()
        res[name] = {"n_patients": int(len(w)), "mean_abs_dHR_bpm": float(d.mean()),
                     "std_abs_dHR_bpm": float(d.std(ddof=1)), "icc_2_1": icc_2_1(w.to_numpy())}
    res["ef_all"] = ef_metrics(df["ef_true"], df["ef_pred"])
    if a.quality_csv:
        import pandas as pd
        df = df.merge(pd.read_csv(a.quality_csv), on="clip_id", how="left")
    if "quality" in df:
        for q, g in df.groupby("quality"):
            res[f"ef_quality_{q}"] = ef_metrics(g["ef_true"], g["ef_pred"])
    save_json(res, a.out)
    print(res)


if __name__ == "__main__":
    main()
