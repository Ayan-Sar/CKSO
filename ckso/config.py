"""YAML configuration with inheritance (``_base_``) and dotted CLI overrides."""

from __future__ import annotations

import copy
import os

import yaml


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("_base_", None)
    if bases:
        if isinstance(bases, str):
            bases = [bases]
        merged = {}
        for b in bases:
            merged = deep_merge(merged, load_config(os.path.join(os.path.dirname(path), b)))
        cfg = deep_merge(merged, cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Overrides like ['train.epochs=3', 'model.operator.kind=dense'] (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got '{item}'")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def save_config(cfg: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
