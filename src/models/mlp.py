"""
Plain MLP baseline for E3 data efficiency comparison.

Directly predicts (u, v, p) from (x, y, t).
No physics constraint — trains on data loss only.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.layers import build_mlp


class MLP(nn.Module):
    """
    Data-driven baseline. Same capacity as PINN for fair comparison.
    """

    def __init__(
        self,
        lb,
        ub,
        hidden_layers: int = 8,
        neurons_per_layer: int = 20,
        activation: str = "tanh",
    ):
        super().__init__()
        lb = np.asarray(lb, dtype=np.float32)
        ub = np.asarray(ub, dtype=np.float32)
        self.register_buffer("lb", torch.from_numpy(lb))
        self.register_buffer("ub", torch.from_numpy(ub))

        self.net = build_mlp(
            in_dim=3, out_dim=3,   # predict (u, v, p) directly
            hidden_layers=hidden_layers,
            neurons_per_layer=neurons_per_layer,
            activation=activation,
        )

    def _normalize(self, x, y, t):
        xyt = torch.cat([x, y, t], dim=1)
        return 2.0 * (xyt - self.lb) / (self.ub - self.lb) - 1.0

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self._normalize(x, y, t)
        out = self.net(z)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p
