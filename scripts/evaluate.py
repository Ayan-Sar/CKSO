"""Evaluate a trained checkpoint on a split; writes predictions.csv and summary.json.

    python scripts/evaluate.py --ckpt runs/echonet/best.pt --split test --out runs/echonet/eval_test
    python scripts/evaluate.py --ckpt ... --protocol central            # Table III
    python scripts/evaluate.py --ckpt ... --readout soft --beta 10       # readout table (Appendix E)
    python scripts/evaluate.py --ckpt ... --perturb speckle8db           # robustness table (Appendix E)
"""

import argparse
import os

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
    p.add_argument("--out", required=True)
    p.add_argument("--protocol", default=None, choices=[None, "all", "three", "central"])
    p.add_argument("--readout", default=None, choices=[None, "roots", "grid", "soft"])
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--perturb", default="none", choices=sorted(PERTURBATIONS))
    p.add_argument("--max-clips", type=int, default=None)
    add_common_args(p)
    a = p.parse_args()
    dev = get_device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    cfg = apply_overrides(cfg, a.override)
    ev = cfg.get("eval", {})
    protocol = a.protocol or ev.get("protocol", "all")
    readout = a.readout or ev.get("readout", "roots")
    beta = a.beta if a.beta is not None else float(ev.get("beta", 40.0))
    pert = PERTURBATIONS[a.perturb](seed=0)
    ds = build_dataset(cfg, a.split, "eval", perturb=pert)
    df, summary = evaluate_dataset(model, ds, protocol, readout, beta, a.max_clips)
    summary.update(ckpt=a.ckpt, split=a.split, protocol=protocol, readout=readout, beta=beta, perturb=a.perturb,
                   params_inference=model.count_parameters(True))
    os.makedirs(a.out, exist_ok=True)
    save_predictions(df, os.path.join(a.out, "predictions.csv"))
    save_json(summary, os.path.join(a.out, "summary.json"))
    print({k: summary[k] for k in ("n_clips", "ef", "hr", "ed_afd", "es_afd") if k in summary})


if __name__ == "__main__":
    main()
