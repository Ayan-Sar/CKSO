"""Structural ablation study (Table V): train + evaluate each variant over seeds.

    python scripts/run_ablations.py --out runs/ablations -o data.root=/data/EchoNet-Dynamic
    python scripts/run_ablations.py --out runs/abl_smoke --base configs/synthetic_smoke.yaml --seeds 0 --device cpu

Per variant and seed: test EF MAE/RMSE, HR MAE (if a reference HR exists), the maximum
over training of max_{s<=64} ||K^s||_2 (logged every epoch), and divergence. The table
written to ablation_table.csv contains only numbers produced by these runs.
"""

import argparse
import json
import os

import pandas as pd

from _common import add_common_args, save_predictions
from ckso.config import apply_overrides, load_config, deep_merge
from ckso.data import build_dataset
from ckso.evaluation import evaluate_dataset
from ckso.training import load_model, train
from ckso.utils import get_device, save_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = {
    "row0_ckso": {},
    "row1_free_frequencies": {"model": {"frequency_mode": "free_subject"}},
    "row2_dense": {"model": {"operator": {"kind": "dense"}}},
    "row3_normal_nonharmonic": {"model": {"frequency_mode": "free_ratio"}},
    "row4_nonnormal": {"model": {"operator": {"normal": False}}},
    "row5_learned_dc": {"model": {"operator": {"learn_dc": True}}},
    "row6_no_freq_loss": {"curriculum": {"weights": {"freq": 0.0}}},
}


def max_norm_from_log(path):
    vals = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("event") == "epoch" and r.get("operator"):
                vals.append(r["operator"]["max_power_norm"])
    return max(vals) if vals else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=os.path.join(ROOT, "configs", "echonet_dynamic.yaml"))
    p.add_argument("--out", required=True)
    p.add_argument("--rows", nargs="*", default=list(VARIANTS))
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--skip-train", action="store_true")
    add_common_args(p)
    a = p.parse_args()
    base = apply_overrides(load_config(a.base), a.override)
    seeds = a.seeds if a.seeds is not None else base["train"].get("seeds", [0])
    dev = get_device(a.device)
    rows = []
    for name in a.rows:
        cfg = deep_merge(base, VARIANTS[name])
        for s in seeds:
            out = os.path.join(a.out, name, f"seed_{s}")
            summ = train(cfg, out, a.device, s) if not a.skip_train else json.load(open(os.path.join(out, "train_summary.json")))
            rec = {"variant": name, "seed": s, "diverged": bool(summ["diverged"]),
                   "max_power_norm_training": max_norm_from_log(os.path.join(out, "train_log.jsonl"))}
            ck = os.path.join(out, "best.pt")
            if os.path.exists(ck) and not summ["diverged"]:
                model, mcfg = load_model(ck, dev)
                ds = build_dataset(mcfg, "test", "eval")
                df, sm = evaluate_dataset(model, ds, progress=False)
                save_predictions(df, os.path.join(out, "eval_test", "predictions.csv"))
                save_json(sm, os.path.join(out, "eval_test", "summary.json"))
                rec.update(ef_mae=sm.get("ef", {}).get("mae"), ef_rmse=sm.get("ef", {}).get("rmse"))
                hr = sm.get("hr")
                rec["hr_mae_bpm"] = hr["learned_omega1"]["hr_mae_bpm"] if isinstance(hr, dict) else float("nan")
            rows.append(rec)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "ablation_runs.csv"), index=False)
    agg = df.groupby("variant").agg(
        ef_mae_mean=("ef_mae", "mean"), ef_mae_std=("ef_mae", "std"), ef_rmse_mean=("ef_rmse", "mean"),
        hr_mae_mean=("hr_mae_bpm", "mean"), max_power_norm=("max_power_norm_training", "max"),
        diverged_seeds=("diverged", "sum"), n_seeds=("seed", "count"))
    agg.to_csv(os.path.join(a.out, "ablation_table.csv"))
    print(agg)


if __name__ == "__main__":
    main()
