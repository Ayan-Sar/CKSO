"""End-to-end: four-stage curriculum on synthetic data, checkpoint, evaluation, scripts."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from ckso.config import apply_overrides, load_config
from ckso.data import build_dataset
from ckso.evaluation import evaluate_dataset
from ckso.training import load_model, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, "configs", "synthetic_smoke.yaml"))
    cfg = apply_overrides(cfg, ["train.log_operator_every=1"])
    out = str(tmp_path_factory.mktemp("run"))
    summary = train(cfg, out, device="cpu", seed=0)
    return root, out, summary


def test_training_runs_all_stages(trained):
    _, out, summary = trained
    assert not summary["diverged"] and summary["epochs_run"] == 4
    logs = [json.loads(l) for l in open(os.path.join(out, "train_log.jsonl"))]
    stages = [r["stage"] for r in logs if r.get("event") == "epoch"]
    assert stages == ["A_autoencode", "B_linearity", "C_horizon", "D_clinical"]
    # ||K^s||_2 == 1 at every logged training step (Proposition 2)
    norms = [r["max_power_norm"] for r in logs if r.get("event") == "step_operator"]
    assert norms and np.allclose(norms, 1.0, atol=1e-6)
    assert os.path.exists(os.path.join(out, "best.pt"))


def test_evaluation_outputs(trained):
    _, out, _ = trained
    model, cfg = load_model(os.path.join(out, "best.pt"))
    ds = build_dataset(cfg, "test", "eval")
    df, summary = evaluate_dataset(model, ds, progress=False)
    assert len(df) == 4 and np.isfinite(df["ef_pred"]).all()
    assert summary["ef"]["n"] == 4 and np.isfinite(summary["ef"]["mae"])
    assert isinstance(summary["hr"], dict)  # synthetic data has a reference HR
    for col in ("lin_residual", "freq_disagreement", "hr_pred_bpm", "hr_periodogram_bpm"):
        assert np.isfinite(df[col]).all()
    for proto in ("central", "three"):
        d2, _ = evaluate_dataset(model, ds, protocol=proto, progress=False)
        assert (d2["n_beats_used"] <= df["n_beats_used"]).all()


def test_cli_scripts(trained, tmp_path):
    root, out, _ = trained
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=root)
    ev = str(tmp_path / "ev")
    subprocess.run([py, os.path.join(root, "scripts", "evaluate.py"), "--ckpt", os.path.join(out, "best.pt"),
                    "--out", ev, "--device", "cpu"], check=True, env=env, capture_output=True)
    assert os.path.exists(os.path.join(ev, "predictions.csv"))
    fa = str(tmp_path / "fa.json")
    subprocess.run([py, os.path.join(root, "scripts", "failure_analysis.py"), "--pred",
                    os.path.join(ev, "predictions.csv"), "--out", fa, "--threshold", "1.0"], check=True, env=env,
                   capture_output=True)
    assert os.path.exists(fa)
    r = subprocess.run([py, os.path.join(root, "scripts", "stability_experiments.py"), "--example"], check=True,
                       env=env, capture_output=True, text=True)
    assert "rho(A) = 0.900" in r.stdout
    r = subprocess.run([py, os.path.join(root, "scripts", "complexity.py"), "--config",
                        os.path.join(root, "configs", "base.yaml"), "--device", "cpu", "--reps", "1"],
                       check=True, env=env, capture_output=True, text=True)
    assert json.loads(r.stdout)["operator_scalars"] == 16
