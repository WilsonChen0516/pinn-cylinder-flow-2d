"""
Training callbacks: checkpointing, periodic validation, snapshot capture.

All callbacks follow a simple interface:
    cb.on_step(state: dict) -> None

where `state` contains at minimum: step, model, optimizer, loss_components.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Base
# -----------------------------------------------------------------------------

class Callback:
    def on_step(self, state: Dict[str, Any]) -> None:  # noqa: D401
        """Called after every training step."""
        pass

    def on_train_end(self, state: Dict[str, Any]) -> None:
        pass


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------

class CheckpointCallback(Callback):
    """Save model + optimizer + metadata every N steps."""

    def __init__(self, out_dir: str | Path, every: int = 5000, keep_last: int = 5):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.keep_last = keep_last
        self._saved_paths: List[Path] = []

    def on_step(self, state: Dict[str, Any]) -> None:
        step = state["step"]
        if step == 0 or step % self.every != 0:
            return
        self._save(state, step)

    def on_train_end(self, state: Dict[str, Any]) -> None:
        # Always save a final checkpoint
        self._save(state, state["step"], suffix="final")

    def _save(self, state: Dict[str, Any], step: int, suffix: str = "") -> None:
        name = f"step_{step:07d}" + (f"_{suffix}" if suffix else "") + ".pt"
        path = self.out_dir / name
        payload = {
            "step": step,
            "model_state_dict": state["model"].state_dict(),
            "optimizer_state_dict": state["optimizer"].state_dict() if state.get("optimizer") else None,
            "loss_components": state.get("loss_components"),
        }
        # Save learnable lambdas for inverse problem
        model = state["model"]
        if getattr(model, "learn_lambdas", False):
            payload["lambda_1"] = float(model.effective_lambda_1.item())
            payload["lambda_2"] = float(model.effective_lambda_2.item())

        torch.save(payload, path)
        self._saved_paths.append(path)

        # Keep only the last `keep_last` regular checkpoints (never delete 'final')
        regular = [p for p in self._saved_paths if "final" not in p.name]
        if len(regular) > self.keep_last:
            to_delete = regular[:-self.keep_last]
            for p in to_delete:
                if p.exists():
                    p.unlink()
                if p in self._saved_paths:
                    self._saved_paths.remove(p)


# -----------------------------------------------------------------------------
# Periodic validation
# -----------------------------------------------------------------------------

class ValidationCallback(Callback):
    """
    Every N steps, run the validator and log metrics.

    Appends to `state['history']['validation']` a dict per evaluation.
    """

    def __init__(self, validator, every: int = 5000, model_type: str = "pinn"):
        self.validator = validator
        self.every = every
        self.model_type = model_type

    def on_step(self, state: Dict[str, Any]) -> None:
        step = state["step"]
        if step == 0 or step % self.every != 0:
            return
        result = self.validator.evaluate(state["model"], model_type=self.model_type)

        entry = {"step": step, **result.as_dict()}
        state.setdefault("history", {}).setdefault("validation", []).append(entry)

        logger = state.get("logger")
        if logger:
            logger.info(
                f"[val @ step {step}]  "
                f"L2(u)={result.relative_l2_u:.4e}  "
                f"L2(v)={result.relative_l2_v:.4e}  "
                f"L2(p)={result.relative_l2_p:.4e}"
            )


# -----------------------------------------------------------------------------
# Track learnable lambdas over training
# -----------------------------------------------------------------------------

class LambdaTrackingCallback(Callback):
    """For inverse problems: record lambda_1, lambda_2 every N steps."""

    def __init__(self, every: int = 100):
        self.every = every

    def on_step(self, state: Dict[str, Any]) -> None:
        step = state["step"]
        if step % self.every != 0:
            return
        model = state["model"]
        if not getattr(model, "learn_lambdas", False):
            return

        entry = {
            "step": step,
            "lambda_1": float(model.effective_lambda_1.item()),
            "lambda_2": float(model.effective_lambda_2.item()),
        }
        state.setdefault("history", {}).setdefault("lambdas", []).append(entry)


# -----------------------------------------------------------------------------
# Loss history tracking (always-on)
# -----------------------------------------------------------------------------

class LossHistoryCallback(Callback):
    """Record loss components every N steps for plotting."""

    def __init__(self, every: int = 100):
        self.every = every

    def on_step(self, state: Dict[str, Any]) -> None:
        step = state["step"]
        if step % self.every != 0:
            return
        comps = state.get("loss_components")
        if comps is None:
            return
        entry = {"step": step, **comps.as_dict()}
        state.setdefault("history", {}).setdefault("loss", []).append(entry)


# -----------------------------------------------------------------------------
# Snapshot: inference field at fixed grid for training animation A2
# -----------------------------------------------------------------------------

class SnapshotCallback(Callback):
    """
    Every N steps, run inference on a fixed grid and save.
    Used to build the training-evolution animation (A2).
    """

    def __init__(
        self,
        grid_x: np.ndarray,  # (N,)
        grid_y: np.ndarray,
        grid_t: float,
        out_dir: str | Path,
        every: int = 2000,
        model_type: str = "pinn",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.model_type = model_type

        # Pre-store as arrays
        self.grid_x = grid_x.astype(np.float32)
        self.grid_y = grid_y.astype(np.float32)
        self.grid_t = float(grid_t)

        self._saved: List[Dict[str, Any]] = []

    def on_step(self, state: Dict[str, Any]) -> None:
        step = state["step"]
        if step % self.every != 0:
            return
        self._capture(state, step)

    def on_train_end(self, state: Dict[str, Any]) -> None:
        self._capture(state, state["step"])
        # Save all snapshots to a single npz
        steps = np.array([s["step"] for s in self._saved])
        u_all = np.stack([s["u"] for s in self._saved])
        v_all = np.stack([s["v"] for s in self._saved])
        np.savez(
            self.out_dir / "snapshots.npz",
            steps=steps, u=u_all, v=v_all,
            grid_x=self.grid_x, grid_y=self.grid_y, grid_t=self.grid_t,
        )

    def _capture(self, state: Dict[str, Any], step: int) -> None:
        model = state["model"]
        device = next(model.parameters()).device

        n = len(self.grid_x)
        x_t = torch.from_numpy(self.grid_x.reshape(-1, 1)).to(device)
        y_t = torch.from_numpy(self.grid_y.reshape(-1, 1)).to(device)
        t_t = torch.full((n, 1), self.grid_t, dtype=torch.float32, device=device)

        model.eval()
        if self.model_type == "pinn":
            x_t = x_t.requires_grad_(True)
            y_t = y_t.requires_grad_(True)
            t_t = t_t.requires_grad_(True)
            from src.physics.navier_stokes import velocity_from_streamfunction
            psi, _ = model(x_t, y_t, t_t)
            u, v = velocity_from_streamfunction(psi, x_t, y_t)
            u = u.detach().cpu().numpy().flatten()
            v = v.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                u, v, _ = model(x_t, y_t, t_t)
                u = u.cpu().numpy().flatten()
                v = v.cpu().numpy().flatten()
        model.train()

        self._saved.append({"step": step, "u": u, "v": v})


# -----------------------------------------------------------------------------
# Persist history + metrics to disk
# -----------------------------------------------------------------------------

def save_history(history: Dict[str, Any], out_dir: str | Path) -> None:
    """Dump history dict to JSON (stringify numpy types)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "history.json"

    def _convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not serializable: {type(o)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, default=_convert, indent=2)


def save_final_metrics(metrics: Dict[str, Any], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"

    def _convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not serializable: {type(o)}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, default=_convert, indent=2)
