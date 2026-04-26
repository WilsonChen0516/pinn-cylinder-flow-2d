"""
PINN network for 2D incompressible NS with stream function formulation.

    Input:  (x, y, t)         shape (M, 3)
    Output: (psi, p)          shape (M, 2)

    u = d(psi)/dy, v = -d(psi)/dx  are computed downstream in physics/.

For the inverse problem, lambda_1 and lambda_2 are exposed as learnable
parameters (via softplus reparametrization for lambda_2 to enforce positivity).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import build_mlp


class PINN(nn.Module):
    """
    Coordinate-input network predicting (psi, p).

    Parameters
    ----------
    lb, ub : array-like, shape (3,)
        Lower and upper bounds for (x, y, t), used for input normalization.
        Inputs are mapped to [-1, 1] before entering the MLP.
    hidden_layers, neurons_per_layer : int
    activation : str
    learn_lambdas : bool
        If True, lambda_1 and lambda_2 are nn.Parameters for inverse problems.
    init_lambda_1, init_lambda_2 : float
        Initial values (for inverse). Only used when learn_lambdas=True.
    """

    def __init__(
        self,
        lb,
        ub,
        hidden_layers: int = 8,
        neurons_per_layer: int = 20,
        activation: str = "tanh",
        learn_lambdas: bool = False,
        init_lambda_1: float = 0.0,
        init_lambda_2: float = 0.0,
    ):
        super().__init__()

        lb = np.asarray(lb, dtype=np.float32)
        ub = np.asarray(ub, dtype=np.float32)
        assert lb.shape == (3,) and ub.shape == (3,), f"lb/ub must be (3,), got {lb.shape}"

        self.register_buffer("lb", torch.from_numpy(lb))
        self.register_buffer("ub", torch.from_numpy(ub))

        self.net = build_mlp(
            in_dim=3, out_dim=2,
            hidden_layers=hidden_layers,
            neurons_per_layer=neurons_per_layer,
            activation=activation,
        )

        self.learn_lambdas = learn_lambdas
        if learn_lambdas:
            # lambda_1 is unconstrained (physically 1.0)
            self.lambda_1 = nn.Parameter(torch.tensor(float(init_lambda_1)))
            # lambda_2 must be positive; use softplus reparameterization
            #   raw_lambda_2 = inverse_softplus(init_lambda_2) if init > 0 else a small value
            if init_lambda_2 > 0:
                raw_init = float(np.log(np.expm1(init_lambda_2)))
            else:
                raw_init = -2.0  # softplus(-2) ~ 0.127
            self.raw_lambda_2 = nn.Parameter(torch.tensor(raw_init))
        else:
            # fixed for forward problem
            self.register_buffer("lambda_1_fixed", torch.tensor(1.0))
            self.register_buffer("lambda_2_fixed", torch.tensor(0.01))

    # -------------------------------------------------------------------------

    @property
    def effective_lambda_1(self) -> torch.Tensor:
        if self.learn_lambdas:
            return self.lambda_1
        return self.lambda_1_fixed

    @property
    def effective_lambda_2(self) -> torch.Tensor:
        if self.learn_lambdas:
            return F.softplus(self.raw_lambda_2)
        return self.lambda_2_fixed

    # -------------------------------------------------------------------------

    def _normalize(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Stack (x, y, t), map to [-1, 1]."""
        xyt = torch.cat([x, y, t], dim=1)
        return 2.0 * (xyt - self.lb) / (self.ub - self.lb) - 1.0

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (psi, p), each shape (M, 1).
        """
        z = self._normalize(x, y, t)
        out = self.net(z)        # (M, 2)
        psi = out[:, 0:1]
        p = out[:, 1:2]
        return psi, p

    # -------------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
