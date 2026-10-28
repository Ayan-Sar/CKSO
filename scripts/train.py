"""Train CKSO (one or several seeds).

    python scripts/train.py --config configs/echonet_dynamic.yaml --out runs/echonet -o data.root=/data/EchoNet-Dynamic
    python scripts/train.py --config configs/synthetic_smoke.yaml --out runs/smoke --device cpu
    python scripts/train.py --config configs/echonet_dynamic.yaml --out runs/echonet --all-seeds
"""

import argparse
import os

from _common import add_common_args  # noqa: F401  (sets sys.path)
from ckso.config import apply_overrides, load_config
from ckso.training import train
from ckso.utils import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--all-seeds", action="store_true", help="train every seed in train.seeds into out/seed_<s>")
    add_common_args(p)
    a = p.parse_args()
    cfg = apply_overrides(load_config(a.config), a.override)
    if a.all_seeds:
        res = []
        for s in cfg["train"].get("seeds", [0]):
            res.append(train(cfg, os.path.join(a.out, f"seed_{s}"), a.device, s))
        save_json(res, os.path.join(a.out, "seeds_summary.json"))
        print(res)
    else:
        print(train(cfg, a.out, a.device, a.seed))


if __name__ == "__main__":
    main()
