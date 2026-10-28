"""Hyper-parameter sweeps (Appendix E): H, delay window m, horizon S, (P, Q).

    python scripts/sweep.py --config configs/echonet_dynamic.yaml --out runs/sweep_H --param model.H --values 1 3 5 7 9
    python scripts/sweep.py --config ... --out runs/sweep_m --param model.m --values 1 2 4 8
    python scripts/sweep.py --config ... --out runs/sweep_S --param horizon --values 1 4 16 32 48

``--param horizon`` sets the final horizon of stages C and D. Note the Nyquist rule:
H with H * omega_1 >= pi cannot occur (omega_max = pi / (H + 1/2)), but a large H shrinks
omega_max and can clip high heart rates; summary.json reports the detector-clamp rate.
"""

import argparse
import copy
import os

import pandas as pd
import yaml

from _common import add_common_args, save_predictions
from ckso.config import apply_overrides, load_config
from ckso.data import build_dataset
from ckso.evaluation import evaluate_dataset
from ckso.training import load_model, train
from ckso.utils import get_device, save_json


def set_horizon(cfg, S):
    cfg = copy.deepcopy(cfg)
    for st in cfg["curriculum"]["stages"]:
        hz = st.get("horizon", 1)
        if isinstance(hz, list):
            st["horizon"] = [min(hz[0], S), S]
        elif "pred" in st["losses"] and st["name"].startswith("D"):
            st["horizon"] = S
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--param", required=True)
    p.add_argument("--values", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=0)
    add_common_args(p)
    a = p.parse_args()
    base = apply_overrides(load_config(a.config), a.override)
    dev = get_device(a.device)
    rows = []
    for v in a.values:
        val = yaml.safe_load(v)
        cfg = set_horizon(base, int(val)) if a.param == "horizon" else apply_overrides(base, [f"{a.param}={v}"])
        out = os.path.join(a.out, f"{a.param}={v}")
        summ = train(cfg, out, a.device, a.seed)
        rec = {a.param: val, "diverged": summ["diverged"]}
        if not summ["diverged"]:
            model, mcfg = load_model(os.path.join(out, "best.pt"), dev)
            df, sm = evaluate_dataset(model, build_dataset(mcfg, "test", "eval"), progress=False)
            save_predictions(df, os.path.join(out, "eval_test", "predictions.csv"))
            save_json(sm, os.path.join(out, "eval_test", "summary.json"))
            rec.update(ef_mae=sm.get("ef", {}).get("mae"), ed_afd=sm["ed_afd"]["mean"], es_afd=sm["es_afd"]["mean"])
            hr = sm.get("hr")
            if isinstance(hr, dict):
                rec["hr_mae_bpm"] = hr["learned_omega1"]["hr_mae_bpm"]
        rows.append(rec)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "sweep.csv"), index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
