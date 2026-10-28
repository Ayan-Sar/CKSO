"""CKSO objective (Section III-E):

L = l_rec L_rec + l_lin L_lin + l_pred L_pred + l_vol L_vol + l_ef L_ef + l_freq L_freq
    + l_perp L_perp + l_pos L_pos + l_var L_var

Definitions (choices not fixed by the manuscript are marked [impl]):
  L_rec  = mean ||x_t - phi(z_t)||^2                      (per-pixel MSE)
  L_lin  = mean_t ||z_{t+1} - K z_t||^2                   (sum over latent dims)
  L_pred = sum_s gamma^{s-1} ||x_{t+s} - phi(K^s z_t)||^2 / sum_s gamma^{s-1}
           [impl] normalised by the weight sum so its scale does not grow with S;
           anchors t drawn uniformly so that t + S stays inside the window.
  L_freq = ((omega_1 - omega_hat) / omega_hat)^2
  L_vol  = ((EDV_beta - EDV)/sigma_V)^2 + ((ESV_beta - ESV)/sigma_V)^2   [impl: scaling]
  L_ef   = ((EF_beta - EF) / ef_scale)^2                  [impl: ef_scale = 10 pp]
  L_pos  = mean relu(-V_n / sigma_V)^2 over the phase grid  (soft positivity)
  L_perp = [impl] mean transverse+scalar energy fraction ||z_perp||^2 / ||z||^2,
           pushing V-relevant content into the harmonic blocks (Theorem 3 hypothesis).
  L_var  = [impl] mean_j relu(1 - std_j(z)), VICReg-style anti-collapse; without it the
           encoder can shrink z and make L_lin trivially small.
Missing labels (NaN) are masked out.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .curriculum import StageState


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return x.new_zeros(())
    return x[mask].mean()


def compute_losses(model, batch: dict, stage: StageState, cfg: dict, generator: torch.Generator | None = None):
    """Returns (total, components dict of floats-as-tensors, aux dict)."""
    w = stage.weights
    frames = batch["frames"]  # (B, T, H, W)
    omega_hat = batch["omega_hat"].to(frames.dtype)
    m = model.m
    z = model.encode(frames)  # (B, T', r)
    B, Tp, r = z.shape
    x_cur = frames[:, m - 1:]  # (B, T', H, W)
    freqs = model.frequencies(z, omega_hat)
    comps = {}
    dev = frames.device

    if "rec" in w:
        xhat = model.decode(z)
        comps["rec"] = F.mse_loss(xhat, x_cur)

    if "lin" in w:
        Kz = model.step(z[:, :-1], freqs)
        comps["lin"] = ((z[:, 1:] - Kz) ** 2).sum(-1).mean()
        if cfg.get("lin_normalize", False):
            # [impl, optional] divide by the (detached) mean latent energy so that growth of
            # the latent scale in stage D cannot let L_lin overpower L_freq and drag omega_1
            # towards 0 (observed on synthetic data; docs/IMPLEMENTATION_NOTES.md section 6).
            comps["lin"] = comps["lin"] / ((z.detach() ** 2).sum(-1).mean() + 1e-8)

    if "pred" in w:
        S = min(stage.horizon, Tp - 1)
        n_anchor = int(cfg.get("n_anchors", 2))
        gamma = float(cfg.get("gamma", 0.98))
        hi = Tp - S  # anchors in [0, hi)
        anchors = torch.randint(0, hi, (B, n_anchor), device="cpu", generator=generator).to(dev)
        z_a = torch.gather(z, 1, anchors[..., None].expand(B, n_anchor, r))
        steps = list(range(1, S + 1))
        zs = model.propagate(z_a, steps, freqs)  # (B, A, S, r)
        xs = model.decode(zs)  # (B, A, S, H, W)
        tgt_idx = anchors[:, :, None] + torch.arange(1, S + 1, device=dev)[None, None]
        tgt = x_cur[torch.arange(B, device=dev)[:, None, None], tgt_idx]  # (B, A, S, H, W)
        per = ((xs - tgt) ** 2).mean((-1, -2))  # (B, A, S)
        gw = gamma ** torch.arange(S, device=dev, dtype=per.dtype)
        comps["pred"] = ((per * gw).sum(-1) / gw.sum()).mean()

    if "freq" in w:
        comps["freq"] = (((freqs["omega1"] - omega_hat) / omega_hat) ** 2).mean()

    if "perp" in w:
        lay = model.layout
        e_all = (z ** 2).sum(-1) + 1e-8
        e_perp = (z[..., lay.trans_slice] ** 2).sum(-1) + (z[..., lay.scal_slice] ** 2).sum(-1)
        comps["perp"] = (e_perp / e_all).mean()

    if "var" in w:
        std = torch.sqrt(z.reshape(-1, r).var(0) + 1e-4)
        comps["var"] = F.relu(1.0 - std).mean()

    if "trace" in w and "frame_target" in batch:
        # [impl, optional] 1 - Pearson(V_t, a_t) against a per-frame surrogate such as the
        # HMC-QU cavity area (scale/offset invariant). Not part of the manuscript's
        # objective; disabled (weight 0 / absent) in all EchoNet configurations.
        tgt = batch["frame_target"][:, m - 1:].to(z)
        ok = torch.isfinite(tgt).all(1)
        if ok.any():
            v = model.volume(z[ok])
            a = tgt[ok]
            v = v - v.mean(1, keepdim=True)
            a = a - a.mean(1, keepdim=True)
            corr = (v * a).sum(1) / (v.norm(dim=1) * a.norm(dim=1) + 1e-8)
            comps["trace"] = (1 - corr).mean()
        else:
            comps["trace"] = z.new_zeros(())

    aux = {"omega1": freqs["omega1"].detach(), "z": z.detach()}
    if stage.clinical:
        stride = int(cfg.get("ref_stride", 4))
        z_ref = z[:, ::stride]
        cl = model.clinical_soft(z_ref, freqs, stage.beta)
        edv = cl["edv"].mean(1)
        esv = cl["esv"].mean(1)
        ef = cl["ef"].mean(1)  # per-reference EF averaged (cf. beat-averaged protocol)
        sigma = model.readout.vol_std
        if "vol" in w:
            tv_ed, tv_es = batch["edv"].to(z), batch["esv"].to(z)
            m_ed, m_es = torch.isfinite(tv_ed), torch.isfinite(tv_es)
            l_ed = _masked_mean(((edv - torch.nan_to_num(tv_ed)) / sigma) ** 2, m_ed)
            l_es = _masked_mean(((esv - torch.nan_to_num(tv_es)) / sigma) ** 2, m_es)
            comps["vol"] = l_ed + l_es
        if "ef" in w:
            tef = batch["ef"].to(z)
            comps["ef"] = _masked_mean(((ef - torch.nan_to_num(tef)) / float(cfg.get("ef_scale", 10.0))) ** 2,
                                       torch.isfinite(tef))
        if "pos" in w:
            comps["pos"] = (F.relu(-cl["V"] / sigma) ** 2).mean()
        aux.update({"ef": ef.detach(), "edv": edv.detach(), "esv": esv.detach()})

    total = sum(w[k] * v for k, v in comps.items())
    return total, comps, aux
