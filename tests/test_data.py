import os

import numpy as np
import pytest
import torch

from ckso.config import load_config
from ckso.data import build_dataset
from ckso.data.base import crop_integer_cycles, normalize_amplitude, train_val_test_split
from ckso.data.echonet_dynamic import disk_volume
from ckso.data.echonet_pediatric import polygon_area
from ckso.data.hmcqu import cavity_area_from_wall
from ckso.data.synthetic import SyntheticSpec, event_frames, volume_curve
from ckso.networks import delay_embed


def test_delay_embedding_alignment():
    x = torch.arange(10, dtype=torch.float32)[None, :, None, None].expand(1, 10, 2, 2)
    X = delay_embed(x, 4)
    assert X.shape == (1, 7, 4, 2, 2)
    assert torch.equal(X[0, 0, :, 0, 0], torch.tensor([0.0, 1, 2, 3]))
    assert torch.equal(X[0, 6, :, 0, 0], torch.tensor([6.0, 7, 8, 9]))  # window j ends at frame j+m-1


def test_crop_integer_cycles():
    assert crop_integer_cycles(100, 30.0, 5) == 90
    assert crop_integer_cycles(20, 30.0, 5) == 20  # less than one cycle: unchanged


def test_normalize_amplitude():
    x = np.random.rand(5, 8, 8).astype(np.float32) * 200
    y = normalize_amplitude(x)
    assert y.min() >= 0 and y.max() <= 1


def test_split_patient_wise_deterministic():
    ids = [f"p{i}" for i in range(100)]
    a = train_val_test_split(ids, seed=3)
    b = train_val_test_split(ids, seed=3)
    assert a == b
    counts = {s: sum(v == s for v in a.values()) for s in ("train", "val", "test")}
    assert counts == {"train": 70, "val": 10, "test": 20}


def test_synthetic_ground_truth_consistent():
    spec = SyntheticSpec(hr_bpm=75.0, edv=150.0, esv=60.0, phi0=1.0, r2=0.25, p2=0.4)
    t = np.linspace(0, spec.n_frames - 1, 50001)
    V = volume_curve(t, spec)
    assert abs(V.max() - 150.0) < 1e-3 and abs(V.min() - 60.0) < 1e-3
    assert abs(spec.ef - 60.0) < 1e-9
    ed, es = event_frames(spec)
    Vf = volume_curve(np.arange(spec.n_frames, dtype=float), spec)
    for f in ed:
        if 1 <= f < spec.n_frames - 1:
            assert Vf[f] >= Vf[f - 1] - 1e-6 or Vf[f] >= Vf[f + 1] - 1e-6


def test_geometry_helpers():
    # disks of a cylinder: long axis 10, 10 chords of diameter 2 -> pi * 1 * 10
    rows = np.array([[0, 0, 0, 10]] + [[-1, y, 1, y] for y in np.linspace(0.5, 9.5, 10)], float)
    assert abs(disk_volume(rows) - np.pi * 10) < 1e-9
    assert abs(polygon_area(np.array([0, 1, 1, 0.0]), np.array([0, 0, 1, 1.0])) - 1.0) < 1e-12
    wall = np.zeros((40, 40), bool)
    wall[10:30, 10:30] = True
    wall[12:28, 12:28] = False
    assert cavity_area_from_wall(wall) == 16 * 16


def test_synthetic_dataset_items(root):
    cfg = load_config(os.path.join(root, "configs", "synthetic_smoke.yaml"))
    ds = build_dataset(cfg, "train", "train")
    it = ds[0]
    assert it["frames"].shape == (40, 112, 112)
    assert 0 <= float(it["frames"].min()) and float(it["frames"].max()) <= 1
    true_w = 2 * np.pi * ds.records[0].hr_bpm / (60 * 50.0)
    assert abs(float(it["omega_hat"]) - true_w) / true_w < 0.15
    ev = build_dataset(cfg, "test", "eval")
    e = ev[0]
    assert e["frames"].shape[0] <= 96 and isinstance(e["ed_frames"], list)
    # train / test splits are disjoint by construction
    assert {r.clip_id for r in ds.records}.isdisjoint({r.clip_id for r in ev.records})


def test_echonet_adapter_on_synthetic_files(tmp_path, root):
    pytest.importorskip("cv2")
    from ckso.data.synthetic import write_echonet_format

    out = write_echonet_format(str(tmp_path / "echo"), (3, 1, 2), seed=1, n_frames=80)
    cfg = load_config(os.path.join(root, "configs", "echonet_dynamic.yaml"))
    cfg["data"]["root"] = out
    cfg["data"]["kwargs"] = {"hr_csv": os.path.join(out, "HeartRate.csv")}
    ds = build_dataset(cfg, "test", "eval")
    assert len(ds) == 2
    rec = ds.records[0]
    assert np.isfinite(rec.ef) and np.isfinite(rec.hr_bpm) and rec.fps == 50.0
    assert len(rec.ed_frames) == 1 and len(rec.es_frames) == 1
    it = ds[0]
    assert it["frames"].ndim == 3 and it["frames"].shape[1:] == (112, 112)
