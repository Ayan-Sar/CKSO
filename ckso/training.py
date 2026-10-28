"""Training loop with the four-stage curriculum (Section III-E, Section IV-A, Appendix D)."""

from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import save_config
from .curriculum import Curriculum
from .data import build_dataset
from .losses import compute_losses
from .model import CKSOModel
from .operator import power_norms
from .utils import JsonlLogger, get_device, save_json, set_seed, to_device


def build_optimizer(model: CKSOModel, opt_cfg: dict) -> torch.optim.Optimizer:
    """AdamW; encoder/decoder/heads at lr_net with weight decay, the operator scalars at
    lr_operator with no weight decay (Section IV-A)."""
    op = model.operator_parameters()
    op_ids = {id(p) for p in op}
    rest = [p for p in model.parameters() if id(p) not in op_ids]
    return torch.optim.AdamW(
        [
            {"params": rest, "lr": float(opt_cfg.get("lr_net", 3e-4)), "weight_decay": float(opt_cfg.get("weight_decay", 1e-2))},
            {"params": op, "lr": float(opt_cfg.get("lr_operator", 1e-3)), "weight_decay": 0.0},
        ],
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
    )


def load_partial_state(model: torch.nn.Module, state: dict) -> list[str]:
    """Load parameters whose names and shapes match (fine-tuning across H / r changes).
    Returns the list of skipped keys."""
    own = model.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(ok)
    model.load_state_dict(own)
    return sorted(set(state) - set(ok))


def load_model(ckpt_path: str, device: torch.device | str = "cpu") -> tuple[CKSOModel, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CKSOModel(ck["config"]["model"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["config"]


@torch.no_grad()
def operator_diagnostics(model: CKSOModel, harm: torch.Tensor, max_power: int = 64) -> dict:
    K = model.operator.matrix(harm[:1].detach().float())[0]
    norms = power_norms(K, max_power)
    eig = torch.linalg.eigvals(K.double()).abs()
    return {"max_power_norm": float(norms.max()), "norm_K": float(norms[0]), "spectral_radius": float(eig.max())}


def _grad_norm(model) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return math.sqrt(tot)


def _make_loader(ds, bs, shuffle, workers, seed):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, drop_last=shuffle and len(ds) >= bs,
                      generator=g, persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


def train(cfg: dict, out_dir: str, device: str = "auto", seed: int | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    tcfg = cfg["train"]
    seed = int(tcfg.get("seed", 0) if seed is None else seed)
    cfg["train"]["seed"] = seed
    cfg["data"]["seed"] = seed
    set_seed(seed, bool(tcfg.get("deterministic", False)))
    dev = get_device(device)
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    log = JsonlLogger(os.path.join(out_dir, "train_log.jsonl"))

    model = CKSOModel(cfg["model"])
    train_ds = build_dataset(cfg, "train", "train")
    vcfg = copy.deepcopy(cfg)
    vcfg["data"]["augment"] = False
    val_ds = build_dataset(vcfg, "val", "train")

    init = tcfg.get("init_checkpoint")
    if init:
        ck = torch.load(init, map_location="cpu", weights_only=False)
        skipped = load_partial_state(model, ck["model"])
        log.log(event="init_checkpoint", path=init, skipped=skipped)
    vm, vs = train_ds.volume_stats(default=(float("nan"), float("nan")))
    if np.isfinite(vm):
        model.set_volume_stats(vm, vs)
    elif not init:
        dm, ds_ = cfg["data"].get("volume_stats_default", [100.0, 40.0])
        model.set_volume_stats(dm, ds_)
    model.to(dev)

    opt = build_optimizer(model, cfg["optim"])
    cur = Curriculum(cfg["curriculum"])
    epochs = int(tcfg.get("epochs", cur.total_epochs))
    bs = int(tcfg.get("batch_size", 16))
    workers = int(tcfg.get("num_workers", 0))
    max_steps = tcfg.get("max_steps_per_epoch")
    clip = float(tcfg.get("grad_clip", 0.0))
    log_every = int(tcfg.get("log_operator_every", 0) or 0)  # per-step ||K^s|| tracking (Section IV-E)
    loss_cfg = cfg.get("loss", {})
    gen = torch.Generator().manual_seed(seed)

    tl = _make_loader(train_ds, bs, True, workers, seed)
    vl = _make_loader(val_ds, bs, False, workers, seed)
    best_metric, best_epoch, diverged = float("inf"), -1, False
    final_stage = len(cur.stages) - 1
    epoch = -1

    for epoch in range(epochs):
        st = cur.state(epoch)
        train_ds.set_epoch(epoch)
        model.train()
        sums, n, t0, gnorms = {}, 0, time.time(), []
        last_harm = None
        for it, batch in enumerate(tl):
            if max_steps and it >= int(max_steps):
                break
            batch = to_device(batch, dev)
            loss, comps, aux = compute_losses(model, batch, st, loss_cfg, gen)
            if not torch.isfinite(loss):
                diverged = True
                log.log(event="diverged", epoch=epoch, step=it, comps={k: float(v) for k, v in comps.items()})
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorms.append(_grad_norm(model))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            with torch.no_grad():
                last_harm = model.frequencies(aux["z"], batch["omega_hat"])["harm"]
            if log_every and it % log_every == 0:
                log.log(event="step_operator", epoch=epoch, step=it, grad_norm=gnorms[-1],
                        **operator_diagnostics(model, last_harm))
            sums["total"] = sums.get("total", 0.0) + float(loss)
            for k, v in comps.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            n += 1
        if diverged:
            break
        train_stats = {k: v / max(n, 1) for k, v in sums.items()}
        diag = operator_diagnostics(model, last_harm) if last_harm is not None else {}

        val = evaluate_losses(model, vl, st, loss_cfg, dev)
        log.log(event="epoch", epoch=epoch, stage=st.name, horizon=st.horizon, beta=st.beta,
                train=train_stats, val=val, operator=diag, grad_norm_mean=float(np.mean(gnorms)) if gnorms else None,
                epoch_time=time.time() - t0)

        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "stage": st.name}
        torch.save(state, os.path.join(out_dir, "last.pt"))
        if st.index == final_stage:
            metric = val.get("ef_mae", val.get("total", float("inf")))
            if metric < best_metric:
                best_metric, best_epoch = metric, epoch
                torch.save(state, os.path.join(out_dir, "best.pt"))
    if best_epoch < 0 and os.path.exists(os.path.join(out_dir, "last.pt")):
        import shutil
        shutil.copyfile(os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt"))
    summary = {"seed": seed, "diverged": diverged, "best_epoch": best_epoch, "best_val_metric": best_metric,
               "epochs_run": epoch + 1 if epochs else 0}
    save_json(summary, os.path.join(out_dir, "train_summary.json"))
    return summary


@torch.no_grad()
def evaluate_losses(model, loader, stage, loss_cfg, dev) -> dict:
    model.eval()
    gen = torch.Generator().manual_seed(12345)
    sums, n, ef_err = {}, 0, []
    for batch in loader:
        batch = to_device(batch, dev)
        loss, comps, aux = compute_losses(model, batch, stage, loss_cfg, gen)
        sums["total"] = sums.get("total", 0.0) + float(loss)
        for k, v in comps.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
        if "ef" in aux:
            e = (aux["ef"] - batch["ef"]).abs()
            ef_err.append(e[torch.isfinite(e)].cpu())
    out = {k: v / max(n, 1) for k, v in sums.items()}
    if ef_err:
        e = torch.cat(ef_err)
        if len(e):
            out["ef_mae"] = float(e.mean())
    model.train()
    return out
