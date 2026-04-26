"""
Trainer: Adam + L-BFGS two-stage training loop.

Design:
- Trainer is agnostic to experiment type (forward/inverse/MLP);
  those variations are encoded in what batches get passed in.
- Batches (obs/col/ic/bc) are prepared once and reused — no mini-batching.
  For 5k-20k collocation points, full-batch is fine on a 3060.
- Adam phase: fixed number of steps with cosine LR decay.
- L-BFGS phase: wraps the same loss function in a closure.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.training.callbacks import Callback


# -----------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,  # PINNLoss or MLPDataLoss
        batches: Dict[str, Dict[str, torch.Tensor]],
        device: str = "cuda",
        callbacks: Optional[List[Callback]] = None,
        logger=None,
    ):
        """
        batches: dict with optional keys 'obs', 'col', 'ic', 'bc'.
                 Each value is itself a dict of tensors (already on device).
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.batches = batches
        self.device = device
        self.callbacks = callbacks or []
        self.logger = logger

        # Running state (mutated during training)
        self.state: Dict[str, Any] = {
            "step": 0,
            "model": self.model,
            "optimizer": None,
            "loss_components": None,
            "history": {"loss": [], "validation": [], "lambdas": []},
            "logger": logger,
        }

    # -------------------------------------------------------------------------
    # Loss wrapper: calls loss_fn with the right batches
    # -------------------------------------------------------------------------

    def _compute_loss(self):
        if hasattr(self.loss_fn, "forward") and "obs" in self.batches and len(self.batches) == 1:
            # MLP-style: data-only
            return self.loss_fn(self.model, obs_batch=self.batches.get("obs"))

        return self.loss_fn(
            self.model,
            obs_batch=self.batches.get("obs"),
            col_batch=self.batches.get("col"),
            ic_batch=self.batches.get("ic"),
            bc_batch=self.batches.get("bc"),
        )

    # -------------------------------------------------------------------------
    # Adam phase
    # -------------------------------------------------------------------------

    def train_adam(
        self,
        n_steps: int,
        lr: float = 1e-3,
        lr_final: float = 1e-4,
        log_every: int = 500,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=lr_final
        )
        self.state["optimizer"] = optimizer

        start_step = self.state["step"]
        t0 = time.time()
        if self.logger:
            self.logger.info(f"=== Adam phase: {n_steps} steps, lr {lr:.1e} -> {lr_final:.1e} ===")

        for i in range(n_steps):
            self.state["step"] = start_step + i + 1

            optimizer.zero_grad()
            loss, comps = self._compute_loss()

            if not torch.isfinite(loss):
                if self.logger:
                    self.logger.error(f"Non-finite loss at step {self.state['step']}: {loss.item()}")
                break

            loss.backward()
            # Gradient clipping guards against occasional spikes
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            self.state["loss_components"] = comps

            # Log
            if self.logger and (i + 1) % log_every == 0:
                elapsed = time.time() - t0
                lr_now = optimizer.param_groups[0]["lr"]
                lam_str = ""
                if getattr(self.model, "learn_lambdas", False):
                    l1 = self.model.effective_lambda_1.item()
                    l2 = self.model.effective_lambda_2.item()
                    lam_str = f"  λ1={l1:.4f}  λ2={l2:.6f}"
                self.logger.info(
                    f"[adam {i+1:6d}/{n_steps}]  "
                    f"total={comps.total:.3e}  "
                    f"data={comps.data:.3e}  pde={comps.pde:.3e}  "
                    f"ic={comps.ic:.3e}  bc={comps.bc:.3e}"
                    f"{lam_str}  lr={lr_now:.2e}  "
                    f"({elapsed:.0f}s)"
                )

            # Callbacks
            for cb in self.callbacks:
                cb.on_step(self.state)

    # -------------------------------------------------------------------------
    # L-BFGS phase
    # -------------------------------------------------------------------------

    def train_lbfgs(
        self,
        max_steps: int = 2000,
        history_size: int = 50,
        max_iter_per_step: int = 20,
        tolerance_grad: float = 1e-8,
        tolerance_change: float = 1e-12,
        log_every: int = 100,
        lr: float = 1.0,
    ):
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=lr,
            max_iter=max_iter_per_step,
            max_eval=int(1.25 * max_iter_per_step),
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn="strong_wolfe",
        )
        self.state["optimizer"] = optimizer

        start_step = self.state["step"]
        t0 = time.time()
        if self.logger:
            self.logger.info(f"=== L-BFGS phase: up to {max_steps} steps ===")

        for i in range(max_steps):
            self.state["step"] = start_step + i + 1

            def closure():
                optimizer.zero_grad()
                loss, comps = self._compute_loss()
                if torch.isfinite(loss):
                    loss.backward()
                self.state["loss_components"] = comps
                return loss

            loss_val = optimizer.step(closure)

            if loss_val is None or not torch.isfinite(torch.as_tensor(loss_val)):
                if self.logger:
                    self.logger.error(f"L-BFGS produced non-finite loss at step {self.state['step']}, stopping")
                break

            if self.logger and (i + 1) % log_every == 0:
                comps = self.state["loss_components"]
                elapsed = time.time() - t0
                lam_str = ""
                if getattr(self.model, "learn_lambdas", False):
                    l1 = self.model.effective_lambda_1.item()
                    l2 = self.model.effective_lambda_2.item()
                    lam_str = f"  λ1={l1:.4f}  λ2={l2:.6f}"
                self.logger.info(
                    f"[lbfgs {i+1:5d}/{max_steps}]  "
                    f"total={comps.total:.3e}  "
                    f"data={comps.data:.3e}  pde={comps.pde:.3e}  "
                    f"ic={comps.ic:.3e}  bc={comps.bc:.3e}"
                    f"{lam_str}  ({elapsed:.0f}s)"
                )

            for cb in self.callbacks:
                cb.on_step(self.state)

    # -------------------------------------------------------------------------

    def on_train_end(self):
        for cb in self.callbacks:
            cb.on_train_end(self.state)