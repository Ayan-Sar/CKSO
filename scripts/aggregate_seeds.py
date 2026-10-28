"""Aggregate summary.json files over seeds into mean +- s.d.

    python scripts/aggregate_seeds.py runs/echonet/seed_*/eval_test/summary.json --out runs/echonet/summary_seeds.json
"""

import argparse
import glob
import json

from _common import ROOT  # noqa: F401
from ckso.stats import aggregate_seeds
from ckso.utils import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summaries", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    files = sorted({f for pat in a.summaries for f in glob.glob(pat)})
    res = [json.load(open(f, encoding="utf-8")) for f in files]
    agg = aggregate_seeds(res)
    agg["_files"] = files
    save_json(agg, a.out)
    print(json.dumps({k: agg[k] for k in ("ef", "ed_afd", "es_afd") if k in agg}, indent=2))


if __name__ == "__main__":
    main()
