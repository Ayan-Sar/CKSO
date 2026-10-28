"""Stability and horizon-cost experiments (Section III-B example, Section IV-E).

  1) --example     the non-normal 2x2 example: rho(A) = 0.9 but ||A^2||_2 >> 1.
  2) --logs DIR... extract max_{s<=64} ||K^s||_2 per epoch / per step from train_log.jsonl
                   (enable per-step logging with -o train.log_operator_every=10 when training).
  3) --benchmark   for each operator variant and horizon S: gradient norm of L_pred w.r.t.
                   the encoder (ratio to S = 1), wall-clock per training step and peak GPU
                   memory, on synthetic clips at initialisation or from --ckpt files.

    python scripts/stability_experiments.py --example
    python scripts/stability_experiments.py --logs runs/ablations/*/seed_0 --out runs/stability/norms.csv
    python scripts/stability_experiments.py --benchmark --horizons 1 4 16 32 48 64 --out runs/stability/cost.csv
"""

import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from _common import ROOT, add_common_args
from ckso.config import apply_overrides, deep_merge, load_config
from ckso.curriculum import StageState
from ckso.data import build_dataset
from ckso.losses import compute_losses
from ckso.model import CKSOModel
from ckso.operator import power_norms
from ckso.utils import get_device, set_seed

BENCH_VARIANTS = {
    "ckso": {},
    "dense": {"model": {"operator": {"kind": "dense"}}},
    "nonnormal_lock": {"model": {"operator": {"normal": False, "init_eta": 0.5}}},
}


def example():
    A = torch.tensor([[0.9, 100.0], [0.0, 0.9]])
    n = power_norms(A, 10)
    rho = float(torch.linalg.eigvals(A.double()).abs().max())
    print(f"rho(A) = {rho:.3f}; ||A^s||_2 for s=1..10:", [round(float(v), 2) for v in n])
    print(f"||A^2||_2 = {float(n[1]):.2f}  (> 180 as stated in Section III-B)")


def logs(dirs, out):
    rows = []
    for d in dirs:
        f = os.path.join(d, "train_log.jsonl")
        if not os.path.exists(f):
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("event") == "epoch" and r.get("operator"):
                rows.append({"run": d, "kind": "epoch", "epoch": r["epoch"], "step": None, **r["operator"]})
            elif r.get("event") == "step_operator":
                rows.append({"run": d, "kind": "step", "epoch": r["epoch"], "step": r["step"],
                             "max_power_norm": r["max_power_norm"], "norm_K": r["norm_K"],
                             "spectral_radius": r["spectral_radius"], "grad_norm": r.get("grad_norm")})
    df = pd.DataFrame(rows)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df.to_csv(out, index=False)
    if len(df):
        print(df.groupby("run")["max_power_norm"].agg(["max", "last"]))
    return df


def benchmark(base_cfg, horizons, variants, ckpts, device, reps, batch_size, out):
    dev = get_device(device)
    cfg_data = copy.deepcopy(base_cfg)
    cfg_data["data"]["dataset"] = "synthetic"
    cfg_data["data"]["augment"] = False
    smax = max(horizons)
    cfg_data["data"]["clip_frames"] = smax + cfg_data["model"].get("m", 4) + 2
    cfg_data["data"]["kwargs"] = {"n_train": batch_size, "seed": 0, "n_frames": max(128, smax + 16), "fps": 50.0}
    ds = build_dataset(cfg_data, "train", "train")
    items = [ds[i] for i in range(batch_size)]
    batch = {k: torch.stack([it[k] for it in items]).to(dev) for k in items[0] if torch.is_tensor(items[0][k])}
    rows = []
    for name in variants:
        set_seed(0)
        cfg = deep_merge(base_cfg, BENCH_VARIANTS[name])
        model = CKSOModel(cfg["model"]).to(dev)
        if name in ckpts:
            model.load_state_dict(torch.load(ckpts[name], map_location=dev, weights_only=False)["model"])
        g1 = None
        for S in horizons:
            st = StageState("bench", 0, 0, {"pred": 1.0}, S, 2.0, False)
            times, gn = [], None
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for r in range(reps + 1):
                model.zero_grad(set_to_none=True)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                loss, _, _ = compute_losses(model, batch, st, {"n_anchors": 1, "gamma": 0.98},
                                            torch.Generator().manual_seed(0))
                loss.backward()
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if r > 0:  # first repetition is warm-up
                    times.append(time.perf_counter() - t0)
                gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.encoder.parameters() if p.grad is not None)))
            if S == min(horizons):
                g1 = gn
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else float("nan")
            rows.append({"variant": name, "S": S, "ms_per_step": 1000 * float(np.median(times)),
                         "peak_mem_gb": mem, "grad_norm_encoder": gn, "grad_ratio_vs_Smin": gn / g1 if g1 else float("nan")})
            print(rows[-1])
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--example", action="store_true")
    p.add_argument("--logs", nargs="*")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 16, 32, 48, 64])
    p.add_argument("--variants", nargs="+", default=list(BENCH_VARIANTS))
    p.add_argument("--ckpt", action="append", default=[], help="variant=path to benchmark trained weights")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="runs/stability/out.csv")
    add_common_args(p)
    a = p.parse_args()
    if a.example:
        example()
    if a.logs:
        logs(sorted({d for pat in a.logs for d in glob.glob(pat)}), a.out)
    if a.benchmark:
        cfg = apply_overrides(load_config(a.config), a.override)
        ck = dict(c.split("=", 1) for c in a.ckpt)
        benchmark(cfg, a.horizons, a.variants, ck, a.device, a.reps, a.batch_size, a.out)


if __name__ == "__main__":
    main()
