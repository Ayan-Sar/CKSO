"""Write a synthetic dataset in the EchoNet-Dynamic file format (Videos/, FileList.csv,
VolumeTracings.csv, plus HeartRate.csv with the true heart rate).

    python scripts/make_synthetic.py --out data/synthetic_echonet --n 64 16 16
    python scripts/train.py --config configs/echonet_dynamic.yaml --out runs/syn_echonet \
        -o data.root=data/synthetic_echonet data.kwargs.hr_csv=data/synthetic_echonet/HeartRate.csv
"""

import argparse

from _common import ROOT  # noqa: F401
from ckso.data.synthetic import write_echonet_format


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n", nargs=3, type=int, default=[64, 16, 16], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--frames", type=int, default=128)
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(write_echonet_format(a.out, tuple(a.n), a.seed, a.frames, a.fps))


if __name__ == "__main__":
    main()
