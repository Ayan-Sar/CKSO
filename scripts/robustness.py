"""Robustness to test-time perturbations (Appendix E, last table).

    python scripts/robustness.py --ckpt runs/echonet/best.pt --out runs/echonet/robustness
    # 50 -> 25 fps on HR > 120 bpm clips needs an H = 3 model (Nyquist) and reference HR:
    python scripts/robustness.py --ckpt runs/echonet_H3/best.pt --perturb downsample2 --min-hr 120 --out ...
"""

import argparse
import os

import numpy as np
import pandas as pd

from _common import add_common_args, save_predictions
from ckso.config import apply_overrides
from ckso.data import build_dataset
from ckso.evaluation import evaluate_dataset
from ckso.perturbations import PERTURBATIONS
from ckso.training import load_model
from ckso.utils import get_device, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--perturb", nargs="+", default=["none", "speckle8db", "blur1.5", "occlusion15", "truncate1"])
    p.add_argument("--min-hr", type=float, default=None, help="restrict to clips with reference HR above this")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--out", required=True)
    add_common_args(p)
    a = p.parse_args()
    dev = get_device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    cfg = apply_overrides(cfg, a.override)
    rows = []
    for name in a.perturb:
        ds = build_dataset(cfg, a.split, "eval", perturb=PERTURBATIONS[name](seed=0))
        if a.min_hr is not None:
            ds.records = [r for r in ds.records if np.isfinite(r.hr_bpm) and r.hr_bpm > a.min_hr]
        df, sm = evaluate_dataset(model, ds, max_clips=a.max_clips, progress=False)
        save_predictions(df, os.path.join(a.out, name, "predictions.csv"))
        save_json(sm, os.path.join(a.out, name, "summary.json"))
        hr = sm.get("hr")
        rows.append({"perturbation": name, "n": sm["n_clips"], "ef_mae": sm.get("ef", {}).get("mae"),
                     "hr_mae_bpm": hr["learned_omega1"]["hr_mae_bpm"] if isinstance(hr, dict) else float("nan")})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "robustness.csv"), index=False)


if __name__ == "__main__":
    main()
