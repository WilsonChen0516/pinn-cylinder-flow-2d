"""
Loss functions for PINN training.

Three kinds of loss terms:
  - L_data : MSE between predicted (u, v) and observations
  - L_pde  : MSE of Navier-Stokes residual (f_u, f_v) against zero
  - L_ic / L_bc : MSE at initial / boundary conditions

The `PINNLoss` class assembles them with configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.physics.navier_stokes import ns_residual_2d, velocity_from_streamfunction


# -----------------------------------------------------------------------------
# Component losses
# -----------------------------------------------------------------------------

def data_loss(u_pred: torch.Tensor, v_pred: torch.Tensor,
              u_true: torch.Tensor, v_true: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and observed (u, v)."""
    return torch.mean((u_pred - u_true) ** 2) + torch.mean((v_pred - v_true) ** 2)


def pde_residual_loss(f_u: torch.Tensor, f_v: torch.Tensor) -> torch.Tensor:
    """MSE of the two momentum residuals."""
    return torch.mean(f_u ** 2) + torch.mean(f_v ** 2)


def mse_to_target(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


# -----------------------------------------------------------------------------
# Weighted loss aggregator for PINN (inverse or forward)
# -----------------------------------------------------------------------------

@dataclass
class LossWeights:
    data: float = 1.0
    pde: float = 1.0
    ic: float = 1.0
    bc: float = 1.0


@dataclass
class LossComponents:
    """Snapshot of the loss terms at a training step (all .item() scalars)."""
    total: float
    data: float = 0.0
    pde: float = 0.0
    ic: float = 0.0
    bc: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"total": self.total, "data": self.data,
                "pde": self.pde, "ic": self.ic, "bc": self.bc}


class PINNLoss(nn.Module):
    """
    Computes PINN loss given the model, observation points, collocation points,
    and optional IC/BC points.

    Supports both forward and inverse modes via the `model.effective_lambda_*`
    properties.
    """

    def __init__(self, weights: Optional[LossWeights] = None):
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self,
        model,
        # --- observation batch (data loss, optional) ---
        obs_batch: Optional[Dict[str, torch.Tensor]] = None,
        # --- collocation batch (PDE loss, required) ---
        col_batch: Optional[Dict[str, torch.Tensor]] = None,
        # --- IC batch (optional) ---
        ic_batch: Optional[Dict[str, torch.Tensor]] = None,
        # --- BC batch (optional) ---
        bc_batch: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Each batch dict must contain 'x', 'y', 't' tensors (requires_grad for col/ic/bc)
        and optionally 'u', 'v' targets (for obs/ic/bc).

        Returns (total_loss, LossComponents).
        """
        w = self.weights
        L_data = torch.tensor(0.0, device=self._device(model))
        L_pde = torch.tensor(0.0, device=self._device(model))
        L_ic = torch.tensor(0.0, device=self._device(model))
        L_bc = torch.tensor(0.0, device=self._device(model))

        # --- Data loss ---
        if obs_batch is not None:
            x, y, t = obs_batch["x"], obs_batch["y"], obs_batch["t"]
            # For PINN: we need psi -> (u, v) via autograd
            x_g = x.detach().requires_grad_(True)
            y_g = y.detach().requires_grad_(True)
            t_g = t.detach().requires_grad_(True)
            psi, _ = model(x_g, y_g, t_g)
            u_pred, v_pred = velocity_from_streamfunction(psi, x_g, y_g)
            L_data = data_loss(u_pred, v_pred, obs_batch["u"], obs_batch["v"])

        # --- PDE residual loss ---
        if col_batch is not None:
            x = col_batch["x"].detach().requires_grad_(True)
            y = col_batch["y"].detach().requires_grad_(True)
            t = col_batch["t"].detach().requires_grad_(True)
            psi, p = model(x, y, t)
            _, _, f_u, f_v = ns_residual_2d(
                psi=psi, p=p, x=x, y=y, t=t,
                lambda_1=model.effective_lambda_1,
                lambda_2=model.effective_lambda_2,
            )
            L_pde = pde_residual_loss(f_u, f_v)

        # --- IC loss ---
        if ic_batch is not None and "u" in ic_batch:
            x = ic_batch["x"].detach().requires_grad_(True)
            y = ic_batch["y"].detach().requires_grad_(True)
            t = ic_batch["t"].detach().requires_grad_(True)
            psi, _ = model(x, y, t)
            u_pred, v_pred = velocity_from_streamfunction(psi, x, y)
            L_ic = data_loss(u_pred, v_pred, ic_batch["u"], ic_batch["v"])

        # --- BC loss ---
        if bc_batch is not None and "u" in bc_batch:
            x = bc_batch["x"].detach().requires_grad_(True)
            y = bc_batch["y"].detach().requires_grad_(True)
            t = bc_batch["t"].detach().requires_grad_(True)
            psi, _ = model(x, y, t)
            u_pred, v_pred = velocity_from_streamfunction(psi, x, y)
            L_bc = data_loss(u_pred, v_pred, bc_batch["u"], bc_batch["v"])

        total = (w.data * L_data + w.pde * L_pde +
                 w.ic * L_ic + w.bc * L_bc)

        comps = LossComponents(
            total=float(total.item()),
            data=float(L_data.item()) if obs_batch is not None else 0.0,
            pde=float(L_pde.item()) if col_batch is not None else 0.0,
            ic=float(L_ic.item()) if (ic_batch is not None and "u" in ic_batch) else 0.0,
            bc=float(L_bc.item()) if (bc_batch is not None and "u" in bc_batch) else 0.0,
        )
        return total, comps

    @staticmethod
    def _device(model) -> torch.device:
        return next(model.parameters()).device


# -----------------------------------------------------------------------------
# Plain MLP loss (E3 baseline)
# -----------------------------------------------------------------------------

class MLPDataLoss(nn.Module):
    """Pure supervised loss for MLP baseline — no PDE term."""

    def forward(self, model, obs_batch: Dict[str, torch.Tensor]):
        x, y, t = obs_batch["x"], obs_batch["y"], obs_batch["t"]
        u_pred, v_pred, _ = model(x, y, t)
        L = data_loss(u_pred, v_pred, obs_batch["u"], obs_batch["v"])
        comps = LossComponents(total=float(L.item()), data=float(L.item()))
        return L, comps
