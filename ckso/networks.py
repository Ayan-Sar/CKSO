"""Encoder psi_theta and decoder phi_eta (Section III-E).

Encoder: four stride-2 3x3 convolutions (GroupNorm, ELU), channels m -> 32 -> 64 -> 96 ->
128 on a 112x112 input, a 1x1 convolution to 32 channels, flatten (32*7*7) and a linear
map to r. The input channels are the m stacked delay frames X_t = (x_{t-m+1}, ..., x_t),
so z_t = psi(X_t) is a time-shared function of the delay window only.

Decoder mirrors it and reconstructs the current frame x_t (one channel, sigmoid output
for amplitude-normalised frames in [0, 1]).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _gn(c: int, groups: int) -> nn.GroupNorm:
    g = groups
    while c % g:
        g -= 1
    return nn.GroupNorm(g, c)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, channels: Sequence[int] = (32, 64, 96, 128),
                 bottleneck: int = 32, image_size: int = 112, groups: int = 8):
        super().__init__()
        if image_size % (2 ** len(channels)):
            raise ValueError(f"image_size must be divisible by {2 ** len(channels)}")
        layers = []
        c_in = in_channels
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), _gn(c, groups), nn.ELU()]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.proj = nn.Conv2d(c_in, bottleneck, 1)
        self.spatial = image_size // (2 ** len(channels))
        self.fc = nn.Linear(bottleneck * self.spatial ** 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (N, m, H, W) -> z (N, r)."""
        h = self.proj(self.features(x))
        return self.fc(h.flatten(1))


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int = 1, channels: Sequence[int] = (32, 64, 96, 128),
                 bottleneck: int = 32, image_size: int = 112, groups: int = 8):
        super().__init__()
        self.spatial = image_size // (2 ** len(channels))
        self.bottleneck = bottleneck
        self.fc = nn.Linear(latent_dim, bottleneck * self.spatial ** 2)
        rev = list(channels)[::-1]  # 128, 96, 64, 32
        self.proj = nn.Sequential(nn.Conv2d(bottleneck, rev[0], 1), _gn(rev[0], groups), nn.ELU())
        layers = []
        for c_in, c_out in zip(rev[:-1], rev[1:]):
            layers += [nn.ConvTranspose2d(c_in, c_out, 3, stride=2, padding=1, output_padding=1),
                       _gn(c_out, groups), nn.ELU()]
        layers += [nn.ConvTranspose2d(rev[-1], out_channels, 3, stride=2, padding=1, output_padding=1)]
        self.up = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z (N, r) -> x (N, 1, H, W) in [0, 1]."""
        h = self.fc(z).view(z.shape[0], self.bottleneck, self.spatial, self.spatial)
        return torch.sigmoid(self.up(self.proj(h)))


def delay_embed(frames: torch.Tensor, m: int) -> torch.Tensor:
    """(B, T, H, W) -> (B, T-m+1, m, H, W); window j ends at frame j+m-1."""
    if frames.shape[1] < m:
        raise ValueError(f"Need at least m={m} frames, got {frames.shape[1]}")
    w = frames.unfold(1, m, 1)  # (B, T', H, W, m)
    return w.permute(0, 1, 4, 2, 3)
