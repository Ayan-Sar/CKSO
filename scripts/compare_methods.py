"""Paired bootstrap comparison with Holm correction (Table II).

Baseline predictions must be *per-video predictions produced by running those methods on
the same test videos* (CSV with an id column and an EF prediction column). Published
summary numbers cannot be used for a paired test.

    python scripts/compare_methods.py --ckso runs/echonet/eval_test/predictions.csv \
        --baseline EchoNet=preds/echonet.csv --baseline CoReEcho=preds/coreecho.csv \
        --id-col FileName --pred-col EF_pred --out runs/echonet/table2.csv
"""

import argparse

import pandas as pd

from _common import load_predictions
from ckso.stats import holm, paired_bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckso", required=True)
    p.add_argument("--baseline", action="append", required=True, help="name=path.csv")
    p.add_argument("--id-col", default="FileName")
    p.add_argument("--pred-col", default="EF_pred")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ck = load_predictions(a.ckso)[["clip_id", "ef_true", "ef_pred"]]
    rows = []
    for item in a.baseline:
        name, path = item.split("=", 1)
        b = pd.read_csv(path)
        b[a.id_col] = b[a.id_col].astype(str).str.replace(".avi", "", regex=False)
        mrg = ck.merge(b[[a.id_col, a.pred_col]], left_on="clip_id", right_on=a.id_col, how="inner")
        r = paired_bootstrap(mrg["ef_true"], mrg["ef_pred"], mrg[a.pred_col], a.n_boot, a.seed)
        rows.append({"comparison": f"CKSO vs {name}", "n": r["n"],
                     "dMAE": r["mae"]["delta"], "dMAE_lo": r["mae"]["ci_low"], "dMAE_hi": r["mae"]["ci_high"],
                     "p_MAE": r["mae"]["p"], "dRMSE": r["rmse"]["delta"], "dRMSE_lo": r["rmse"]["ci_low"],
                     "dRMSE_hi": r["rmse"]["ci_high"], "p_RMSE": r["rmse"]["p"]})
    df = pd.DataFrame(rows)
    # Holm over the whole family of tests (both metrics, all baselines).
    adj = holm(list(df["p_MAE"]) + list(df["p_RMSE"]))
    df["p_MAE_holm"] = adj[: len(df)]
    df["p_RMSE_holm"] = adj[len(df):]
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
