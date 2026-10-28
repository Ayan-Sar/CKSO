import os

import pytest
import torch

from ckso.config import apply_overrides, deep_merge, load_config
from ckso.curriculum import Curriculum, StageState
from ckso.losses import compute_losses
from ckso.model import CKSOModel


@pytest.fixture
def cfg(root):
    return load_config(os.path.join(root, "configs", "synthetic_smoke.yaml"))


def fake_batch(B=2, T=24, S=112, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"frames": torch.rand(B, T, S, S, generator=g), "omega_hat": torch.tensor([0.2, 0.3])[:B],
            "ef": torch.tensor([55.0, float("nan")])[:B], "edv": torch.tensor([120.0, 90.0])[:B],
            "esv": torch.tensor([50.0, float("nan")])[:B], "frame_target": torch.full((B, T), float("nan"))}


def test_encoder_shapes_and_params(cfg):
    m = CKSOModel(cfg["model"])
    assert m.encoder.spatial == 7 and m.encoder.fc.in_features == 32 * 7 * 7
    z = m.encode(torch.rand(2, 10, 112, 112))
    assert z.shape == (2, 7, 22)
    x = m.decode(z)
    assert x.shape == (2, 7, 112, 112) and float(x.min()) >= 0 and float(x.max()) <= 1
    assert m.count_parameters(True) < m.count_parameters(False)


def test_chunked_encoding_matches(cfg):
    m = CKSOModel(cfg["model"]).eval()
    x = torch.rand(1, 20, 112, 112)
    with torch.no_grad():
        assert torch.allclose(m.encode(x), m.encode(x, chunk=5), atol=1e-5)


VARIANTS = {
    "ckso": {},
    "free_subject": {"model": {"frequency_mode": "free_subject"}},
    "free_ratio": {"model": {"frequency_mode": "free_ratio"}},
    "dense": {"model": {"operator": {"kind": "dense"}}},
    "nonnormal": {"model": {"operator": {"normal": False, "init_eta": 0.3}}},
    "learn_dc": {"model": {"operator": {"learn_dc": True}}},
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_all_variants_losses_finite_and_backward(cfg, name):
    c = deep_merge(cfg, VARIANTS[name])
    m = CKSOModel(c["model"])
    m.set_volume_stats(90.0, 30.0)
    st = StageState("D", 3, 0, dict(c["curriculum"]["weights"]), 6, 10.0, True)
    st.weights.pop("trace")
    loss, comps, aux = compute_losses(m, fake_batch(), st, c["loss"], torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    assert set(comps) == {"rec", "lin", "pred", "vol", "ef", "freq", "perp", "pos", "var"}
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    expected = {"ckso": "analytic", "free_subject": "general", "free_ratio": "general", "dense": "trace",
                "nonnormal": "trace", "learn_dc": "analytic"}[name]
    assert m.readout_mode == expected
    with torch.no_grad():
        z = m.encode(fake_batch()["frames"])
        f = m.frequencies(z, torch.tensor([0.2, 0.3]))
        for meth in ("roots", "grid", "soft"):
            out = m.clinical_exact(z[:, 0], f, method=meth)
            assert torch.isfinite(out["ef"]).all()


def test_harmonic_lock_frequencies(cfg):
    m = CKSOModel(cfg["model"])
    z = torch.randn(3, 9, m.r)
    f = m.frequencies(z, torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(f["harm"], f["omega1"][:, None] * torch.arange(1, 6))
    assert torch.allclose(f["omega1"], torch.tensor([0.1, 0.2, 0.3]), atol=1e-5)


def test_clinical_soft_vs_exact_close_at_high_beta(cfg):
    c = apply_overrides(cfg, ["model.n_phase_grid=1024"])
    m = CKSOModel(c["model"]).double()
    m.set_volume_stats(100.0, 30.0)
    z = torch.randn(4, 1, m.r, dtype=torch.float64)
    f = {"omega1": torch.full((4,), 0.2, dtype=torch.float64), "harm": 0.2 * torch.arange(1, 6, dtype=torch.float64).expand(4, 5)}
    s = m.clinical_soft(z, f, beta=1e4)
    e = m.clinical_exact(z[:, 0], f, method="roots")
    assert torch.allclose(s["edv"][:, 0], e["edv"], atol=1e-2)
    assert torch.allclose(s["esv"][:, 0], e["esv"], atol=1e-2)


def test_curriculum_schedule(root):
    cfg = load_config(os.path.join(root, "configs", "base.yaml"))
    cur = Curriculum(cfg["curriculum"])
    assert cur.total_epochs == 100
    a = cur.state(0)
    assert a.name.startswith("A") and set(a.weights) == {"rec", "var"} and not a.clinical
    b = cur.state(15)
    assert b.name.startswith("B") and b.horizon == 2 and "lin" in b.weights and "freq" in b.weights
    c0, c1 = cur.state(40), cur.state(69)
    assert c0.horizon == 2 and c1.horizon == 48
    mid = cur.state(55).horizon
    assert 2 < mid < 48
    d0, d1 = cur.state(70), cur.state(99)
    assert d0.clinical and abs(d0.beta - 2.0) < 1e-9 and abs(d1.beta - 40.0) < 1e-9
    assert d1.weights == {"rec": 1.0, "var": 0.05, "lin": 2.0, "freq": 0.5, "perp": 0.1, "pred": 1.0,
                          "vol": 5.0, "ef": 10.0, "pos": 1.0}
    assert cur.state(500).name == d1.name  # clamps past the end


def test_config_inheritance_and_overrides(root):
    cfg = load_config(os.path.join(root, "configs", "ablations", "row2_dense.yaml"))
    assert cfg["model"]["operator"]["kind"] == "dense" and cfg["model"]["H"] == 5
    assert cfg["data"]["dataset"] == "echonet_dynamic"
    cfg2 = apply_overrides(cfg, ["model.H=3", "train.epochs=2"])
    assert cfg2["model"]["H"] == 3 and cfg2["train"]["epochs"] == 2 and cfg["model"]["H"] == 5
    hm = load_config(os.path.join(root, "configs", "hmcqu.yaml"))
    assert (hm["model"]["H"], hm["model"]["P"], hm["model"]["Q"]) == (3, 3, 2)
    assert hm["curriculum"]["weights"]["trace"] == 5.0 and hm["curriculum"]["weights"]["ef"] == 10.0
