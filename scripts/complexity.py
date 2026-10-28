"""Parameter count, analytic GFLOPs and latency (Section IV-B efficiency paragraph).

    python scripts/complexity.py --config configs/echonet_dynamic.yaml --frames 64 --device cuda
    python scripts/complexity.py --config configs/echonet_dynamic.yaml --device cpu --threads 1
"""

import argparse
import json

import torch

from _common import add_common_args
from ckso.complexity import model_complexity
from ckso.config import apply_overrides, load_config
from ckso.model import CKSOModel
from ckso.utils import get_device


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--frames", type=int, default=64)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--reps", type=int, default=20)
    add_common_args(p)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = apply_overrides(load_config(a.config), a.override)
    model = CKSOModel(cfg["model"])
    dev = str(get_device(a.device))
    print(json.dumps(model_complexity(model, a.frames, dev, a.reps), indent=2))


if __name__ == "__main__":
    main()
