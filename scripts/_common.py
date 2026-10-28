"""Shared CLI helpers (scripts run without installing the package)."""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

LIST_COLUMNS = ("ef_pred_beats", "ed_events", "es_events", "ed_afd", "es_afd", "alpha", "rho_emp", "volume_trace")


def save_predictions(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = df.copy()
    for c in LIST_COLUMNS:
        if c in out:
            out[c] = out[c].map(json.dumps)
    out.to_csv(path, index=False)


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in LIST_COLUMNS:
        if c in df:
            df[c] = df[c].map(lambda s: json.loads(s) if isinstance(s, str) else [])
    return df


def add_common_args(p):
    p.add_argument("--device", default="auto")
    p.add_argument("--override", "-o", nargs="*", default=[], help="dotted overrides, e.g. data.root=/data/echonet")
    return p
