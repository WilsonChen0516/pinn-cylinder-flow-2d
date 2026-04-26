"""
Validation: compare PINN predictions against Raissi ground truth
at a fixed grid of points, compute metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from src.data.loader import CylinderWakeDataset
from src.evaluation.metrics import relative_l2, rmse


@dataclass
class ValidationResult:
    relative_l2_u: float
    relative_l2_v: float
    relative_l2_p: float  # pressure compared after mean subtraction
    rmse_u: float
    rmse_v: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "relative_l2_u": self.relative_l2_u,
            "relative_l2_v": self.relative_l2_v,
            "relative_l2_p": self.relative_l2_p,
            "rmse_u": self.rmse_u,
            "rmse_v": self.rmse_v,
        }


class Validator:
    """
    Build a validation set from the full ground truth dataset,
    run inference, and compute metrics.
    """

    def __init__(
        self,
        dataset: CylinderWakeDataset,
        device: str = "cuda",
        max_points: int = 20000,
    ):
        self.device = device

        total = dataset.total_points()
        if total > max_points:
            rng = np.random.default_rng(9999)  # fixed seed for consistent eval
            idx = rng.choice(total, size=max_points, replace=False)
            self.x = dataset.x[idx]
            self.y = dataset.y[idx]
            self.t = dataset.t[idx]
            self.u_true = dataset.u[idx]
            self.v_true = dataset.v[idx]
            self.p_true = dataset.p[idx]
        else:
            self.x = dataset.x
            self.y = dataset.y
            self.t = dataset.t
            self.u_true = dataset.u
            self.v_true = dataset.v
            self.p_true = dataset.p

        # Preload to device as float32 tensors (no grad needed for validation)
        self._x = torch.from_numpy(self.x).to(device)
        self._y = torch.from_numpy(self.y).to(device)
        self._t = torch.from_numpy(self.t).to(device)

    @torch.no_grad()
    def _infer_mlp(self, model) -> Dict[str, np.ndarray]:
        u, v, p = model(self._x, self._y, self._t)
        return {
            "u": u.cpu().numpy(),
            "v": v.cpu().numpy(),
            "p": p.cpu().numpy(),
        }

    def _infer_pinn(self, model) -> Dict[str, np.ndarray]:
        """PINN inference requires grad to get u, v from psi (via autograd)."""
        from src.physics.navier_stokes import velocity_from_streamfunction

        x = self._x.clone().requires_grad_(True)
        y = self._y.clone().requires_grad_(True)
        t = self._t.clone().requires_grad_(True)

        psi, p = model(x, y, t)
        u, v = velocity_from_streamfunction(psi, x, y)

        return {
            "u": u.detach().cpu().numpy(),
            "v": v.detach().cpu().numpy(),
            "p": p.detach().cpu().numpy(),
        }

    def evaluate(self, model, model_type: str = "pinn") -> ValidationResult:
        """
        model_type: "pinn" (output psi, p) or "mlp" (output u, v, p directly).
        """
        model.eval()
        if model_type == "pinn":
            preds = self._infer_pinn(model)
        elif model_type == "mlp":
            preds = self._infer_mlp(model)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        model.train()

        # Pressure only defined up to a constant — subtract means
        p_pred = preds["p"] - preds["p"].mean()
        p_true = self.p_true - self.p_true.mean()

        return ValidationResult(
            relative_l2_u=relative_l2(preds["u"], self.u_true),
            relative_l2_v=relative_l2(preds["v"], self.v_true),
            relative_l2_p=relative_l2(p_pred, p_true),
            rmse_u=rmse(preds["u"], self.u_true),
            rmse_v=rmse(preds["v"], self.v_true),
        )
