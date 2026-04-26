"""
Static plots for reports.

- Field comparison (ground truth / prediction / error)
- Loss curves
- Data efficiency (E3 core plot)
- Parameter identification curves (λ₁, λ₂)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.styles import apply_style, CMAP_VELOCITY, CMAP_PRESSURE, CMAP_ERROR

apply_style()


def plot_field_comparison(
    x: np.ndarray,
    y: np.ndarray,
    gt: Dict[str, np.ndarray],        # {'u': ..., 'v': ..., 'p': ...}, each shape (N,)
    pred: Dict[str, np.ndarray],
    out_path: Path,
    *,
    time_label: str | None = None,
) -> None:
    """3x3 grid: rows = (u, v, p), cols = (GT, pred, |error|)."""
    fields = ["u", "v", "p"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)

    for i, f in enumerate(fields):
        g = gt[f].copy()
        p = pred[f].copy()

        if f == "p":
            # Pressure only defined up to a constant — center both
            g = g - g.mean()
            p = p - p.mean()
            cmap = CMAP_PRESSURE
        else:
            cmap = CMAP_VELOCITY

        err = np.abs(g - p)

        # Shared color range for GT and prediction
        vmax = float(max(np.abs(g).max(), np.abs(p).max()))
        if f == "p":
            # Pressure can be asymmetric; use actual min/max
            vmin = float(min(g.min(), p.min()))
            vmax = float(max(g.max(), p.max()))
        else:
            vmin = -vmax

        # GT
        cs0 = axes[i, 0].tricontourf(x, y, g, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Ground truth — {f}")
        axes[i, 0].set_aspect("equal")
        fig.colorbar(cs0, ax=axes[i, 0], shrink=0.7)

        # Pred
        cs1 = axes[i, 1].tricontourf(x, y, p, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"PINN — {f}")
        axes[i, 1].set_aspect("equal")
        fig.colorbar(cs1, ax=axes[i, 1], shrink=0.7)

        # Error
        cs2 = axes[i, 2].tricontourf(x, y, err, levels=30, cmap=CMAP_ERROR)
        axes[i, 2].set_title(f"|error| — {f}")
        axes[i, 2].set_aspect("equal")
        fig.colorbar(cs2, ax=axes[i, 2], shrink=0.7)

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if time_label:
        fig.suptitle(time_label, fontsize=14)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_loss_curves(
    history: Dict[str, Sequence[float]],
    out_path: Path,
    *,
    log_y: bool = True,
) -> None:
    """
    Plot multiple loss components over training steps.

    history keys may include: 'step', 'total', 'data', 'pde', 'ic', 'bc'
    (only 'step' and one or more of the others are required).
    """
    assert "step" in history, "history must contain 'step' key"
    steps = np.asarray(history["step"])

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for key, values in history.items():
        if key == "step":
            continue
        ax.plot(steps, values, label=key.upper())

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Training loss curves")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_lambda_convergence(
    steps: Sequence[int],
    lambda_1: Sequence[float],
    lambda_2: Sequence[float],
    true_lambda_1: float,
    true_lambda_2: float,
    out_path: Path,
    *,
    x_log: bool = False,
    x_max: float | None = None,
) -> None:
    """Plot identified λ₁, λ₂ vs training step.

    Parameters
    ----------
    x_log : if True, use log scale on x-axis (old default).
            Default False = linear scale (better for showing transient
            features in early training).
    x_max : if given, sets right limit of x-axis. Useful to crop tail
            where λ has converged and nothing interesting happens.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(steps, lambda_1, label="identified")
    axes[0].axhline(true_lambda_1, ls="--", c="red", label=f"true ({true_lambda_1})")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel(r"$\lambda_1$")
    axes[0].set_title(r"$\lambda_1$ convergence")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    if x_log:
        axes[0].set_xscale("log")
    if x_max is not None:
        axes[0].set_xlim(left=0, right=x_max)

    axes[1].plot(steps, lambda_2, label="identified")
    axes[1].axhline(true_lambda_2, ls="--", c="red", label=f"true ({true_lambda_2})")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel(r"$\lambda_2$")
    axes[1].set_title(r"$\lambda_2$ convergence")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    if x_log:
        axes[1].set_xscale("log")
    if x_max is not None:
        axes[1].set_xlim(left=0, right=x_max)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_data_efficiency(
    n_points_list: Sequence[int],
    pinn_errors: Sequence[float],
    mlp_errors: Dict[int, float],   # {n_points: error}
    out_path: Path,
    *,
    metric_name: str = "Relative L2 error (u)",
) -> None:
    """
    E3 core plot: x = log(N), y = error, two lines (PINN vs MLP).
    """
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    # PINN curve
    ax.plot(n_points_list, pinn_errors, "o-", label="PINN (with physics)", linewidth=2, markersize=8)

    # MLP points
    mlp_n = sorted(mlp_errors.keys())
    mlp_e = [mlp_errors[n] for n in mlp_n]
    ax.plot(mlp_n, mlp_e, "s--", label="MLP (data only)", linewidth=2, markersize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of observation points")
    ax.set_ylabel(metric_name)
    ax.set_title("Data efficiency: PINN vs. pure data-driven MLP")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")