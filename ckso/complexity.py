"""Parameter / FLOP / latency accounting (Section IV-B efficiency paragraph).

FLOPs are counted analytically with forward hooks on Conv2d / ConvTranspose2d / Linear
(multiply-accumulates, MACs); GFLOPs are reported as 2 x MACs, the usual convention.
Normalisation, activation and operator FLOPs are listed separately.
"""

from __future__ import annotations

import time

import torch
from torch import nn


def count_macs(module: nn.Module, inputs: torch.Tensor) -> int:
    macs = [0]

    def conv_hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1]
        if isinstance(m, nn.ConvTranspose2d):
            # every input element is scattered into k * C_out/groups outputs
            macs[0] += i[0].numel() * (m.out_channels // m.groups) * k
        else:
            macs[0] += o.numel() * (m.in_channels // m.groups) * k

    def lin_hook(m, i, o):
        macs[0] += o.numel() * m.in_features

    hs = []
    for mod in module.modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            hs.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hs.append(mod.register_forward_hook(lin_hook))
    with torch.no_grad():
        module(inputs)
    for h in hs:
        h.remove()
    return macs[0]


def operator_flops(layout) -> int:
    """FLOPs of one application of a pre-assembled K: each 2x2 block is a 2x2 mat-vec
    (4 mult + 2 add), scalars 1 mult each, the DC block is free. Forming rho*cos and
    rho*sin is a once-per-clip cost and is not included."""
    blocks = layout.H + layout.P
    return blocks * (4 + 2) + layout.Q


@torch.no_grad()
def model_complexity(model, n_frames: int = 64, device: str = "cpu", latency_reps: int = 10) -> dict:
    model = model.to(device).eval()
    S = model.image_size
    win = torch.zeros(n_frames - model.m + 1, model.m, S, S, device=device)
    enc_macs = count_macs(model.encoder, win)
    head_macs = 0
    for mod in model.freq_head.modules():
        if isinstance(mod, nn.Linear):
            head_macs += mod.in_features * mod.out_features
    frames = torch.rand(1, n_frames, S, S, device=device)
    wh = torch.tensor([0.3], device=device)

    def run():
        z = model.encode(frames)
        f = model.frequencies(z, wh)
        model.clinical_exact(z[0, :1], {k: v[:1] for k, v in f.items()})

    run()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(latency_reps):
        run()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / latency_reps
    return {
        "n_frames": n_frames,
        "params_inference": model.count_parameters(True),
        "params_total_incl_decoder": model.count_parameters(False),
        "params_encoder": sum(p.numel() for p in model.encoder.parameters()),
        "operator_scalars": sum(p.numel() for p in model.operator_parameters()),
        "encoder_macs": int(enc_macs),
        "gflops_inference": 2.0 * (enc_macs + head_macs) / 1e9,
        "operator_flops_per_application": operator_flops(model.layout),
        "latency_ms": 1000.0 * lat,
        "device": device,
    }
