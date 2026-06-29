"""The velocity field network v_theta(x, t).

A small MLP that predicts the flow velocity at a point ``x`` and time ``t``.
Two of the planned architecture upgrades live here: a **Fourier feature** time
embedding (so the network can represent the fast time-dependence of the velocity
near t=0/1) and **GELU** activations.
"""

import math

import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    """Random Fourier features for the scalar time t -> (B, dim).

    Frequencies are drawn once and fixed (registered as a buffer, not trained),
    so the embedding is deterministic for a given seed and moves with ``.to()``.
    """

    def __init__(self, dim=64, scale=10.0, seed=0):
        super().__init__()
        assert dim % 2 == 0, "time embedding dim must be even"
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(dim // 2, generator=g) * scale
        self.register_buffer("w", w)

    def forward(self, t):
        t = t.reshape(-1, 1)                 # (B, 1)
        proj = 2.0 * math.pi * t * self.w    # (B, dim/2)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class VelocityField(nn.Module):
    """MLP velocity field. ``forward(x, t)`` returns the velocity at (x, t)."""

    def __init__(self, dim, hidden=128, depth=3, time_dim=64, seed=0):
        super().__init__()
        self.time_emb = FourierTimeEmbedding(time_dim, seed=seed)
        layers = []
        in_dim = dim + time_dim
        for _ in range(depth):
            layers += [nn.Linear(in_dim, hidden), nn.GELU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        te = self.time_emb(t)
        return self.net(torch.cat([x, te], dim=-1))
