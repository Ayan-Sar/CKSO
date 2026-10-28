"""Failure prediction and abstention (Section V, "Failure as a feature").

Uses quantities already in predictions.csv: the per-clip linearity residual L_lin and
the frequency disagreement |omega_1 - omega_hat| / omega_hat.
  * AUC of each for detecting |EF error| > threshold (default 10 pp);
  * combined score: logistic regression on (log L_lin, log disagreement), out-of-fold
    predictions from stratified K-fold so the AUC is not optimistic;
  * abstention: MAE after removing the top q% highest-residual clips;
  * optional subgroup analysis via --subgroup-csv (clip_id, group), e.g. sinus vs AF/ectopy;
    rhythm labels are not part of EchoNet-Dynamic and must be supplied by the user.

    python scripts/failure_analysis.py --pred runs/echonet/eval_test/predictions.csv --out runs/echonet/failure.json
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from _common import load_predictions
from ckso.metrics import ef_metrics
from ckso.utils import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True)
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--abstain", nargs="*", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--subgroup-csv", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    df = load_predictions(a.pred)
    df = df[np.isfinite(df["ef_true"]) & np.isfinite(df["ef_pred"])].reset_index(drop=True)
    err = (df["ef_pred"] - df["ef_true"]).abs().to_numpy()
    y = (err > a.threshold).astype(int)
    res = {"n": int(len(df)), "n_failures": int(y.sum()), "threshold_pp": a.threshold}
    feats = np.stack([np.log(df["lin_residual"].to_numpy() + 1e-12),
                      np.log(df["freq_disagreement"].to_numpy() + 1e-12)], 1)
    if 0 < y.sum() < len(y):
        res["auc_lin_residual"] = float(roc_auc_score(y, feats[:, 0]))
        res["auc_freq_disagreement"] = float(roc_auc_score(y, feats[:, 1]))
        k = int(min(5, y.sum(), len(y) - y.sum()))
        if k >= 2:
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(feats, y):
                mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-12
                clf = LogisticRegression().fit((feats[tr] - mu) / sd, y[tr])
                oof[te] = clf.predict_proba((feats[te] - mu) / sd)[:, 1]
            res["auc_combined_oof"] = float(roc_auc_score(y, oof))
    else:
        res["note"] = "AUC undefined: only one class present"
    order = np.argsort(-df["lin_residual"].to_numpy())
    res["mae_all"] = ef_metrics(df["ef_true"], df["ef_pred"])["mae"]
    res["abstention"] = {}
    for q in a.abstain:
        n_drop = int(round(q * len(df)))
        keep = np.sort(order[n_drop:])
        res["abstention"][f"{q:.2f}"] = {"n_kept": int(len(keep)),
                                         "mae": ef_metrics(df["ef_true"][keep], df["ef_pred"][keep])["mae"],
                                         "dropped_ids": df["clip_id"].iloc[order[:n_drop]].tolist()}
    if a.subgroup_csv:
        sg = pd.read_csv(a.subgroup_csv)
        m = df.merge(sg, on="clip_id", how="inner")
        res["subgroups"] = {}
        for g, gg in m.groupby("group"):
            res["subgroups"][str(g)] = {**ef_metrics(gg["ef_true"], gg["ef_pred"]),
                                        "lin_residual_median": float(gg["lin_residual"].median()),
                                        "freq_disagreement_median": float(gg["freq_disagreement"].median())}
        for q in a.abstain:
            dropped = set(res["abstention"][f"{q:.2f}"]["dropped_ids"])
            dm = m[m["clip_id"].isin(dropped)]
            res["abstention"][f"{q:.2f}"]["dropped_group_fractions"] = dm["group"].value_counts(normalize=True).to_dict()
    save_json(res, a.out)
    print({k: v for k, v in res.items() if k != "abstention"})


if __name__ == "__main__":
    main()
