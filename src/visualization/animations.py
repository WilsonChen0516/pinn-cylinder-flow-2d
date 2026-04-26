"""
Animations for reports and README.

Three animations:
    A1 — Ground truth vs PINN evolution (side by side)
    A2 — Training evolution (PINN output over training steps)
    A3 — Sparse reconstruction (scatter obs -> full field)

Output: mp4 (via ffmpeg) or gif (via Pillow, fallback).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim

from src.visualization.styles import apply_style, CMAP_VORTICITY, CMAP_ERROR

apply_style()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _save_animation(anim: manim.FuncAnimation, out_path: Path, fps: int = 20) -> None:
    """Save as mp4 if ffmpeg available, else gif."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    if ext == ".mp4":
        try:
            writer = manim.FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(out_path, writer=writer)
            print(f"  saved: {out_path}")
            return
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  ffmpeg unavailable ({e}); falling back to gif")
            out_path = out_path.with_suffix(".gif")
            ext = ".gif"

    if ext == ".gif":
        writer = manim.PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
        print(f"  saved: {out_path}")
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def _tricontour_axis(
    ax, x: np.ndarray, y: np.ndarray, field: np.ndarray, vmin: float, vmax: float,
    cmap: str, title: str
):
    """Draw one tricontour plot. Returns the QuadContourSet for later update."""
    cs = ax.tricontourf(x, y, field, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return cs


# -----------------------------------------------------------------------------
# A1: Ground truth vs PINN evolution (vorticity side-by-side)
# -----------------------------------------------------------------------------

def animate_field_evolution(
    x: np.ndarray,
    y: np.ndarray,
    gt_series: np.ndarray,       # (T, N) ground truth field over time
    pred_series: np.ndarray,     # (T, N) prediction field over time, or None for GT-only
    t_values: np.ndarray,        # (T,)
    out_path: Path,
    *,
    field_name: str = "vorticity",
    cmap: str = CMAP_VORTICITY,
    fps: int = 20,
    symmetric_colorbar: bool = True,
) -> None:
    """
    Side-by-side animation of a scalar field over time.

    If pred_series is None, shows ground truth only (useful for Day 1 sanity check).
    """
    T = gt_series.shape[0]
    assert len(t_values) == T
    pred_given = pred_series is not None

    if symmetric_colorbar:
        vmax = float(np.abs(gt_series).max())
        vmin = -vmax
    else:
        vmin = float(gt_series.min())
        vmax = float(gt_series.max())

    ncols = 3 if pred_given else 1  # GT / Pred / |error|
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(5 * ncols, 4.5),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]

    # Initial frame
    cs_gt = _tricontour_axis(axes[0], x, y, gt_series[0], vmin, vmax, cmap,
                             f"Ground truth — {field_name}")
    fig.colorbar(cs_gt, ax=axes[0], shrink=0.7)

    if pred_given:
        cs_pred = _tricontour_axis(axes[1], x, y, pred_series[0], vmin, vmax, cmap,
                                   f"PINN — {field_name}")
        fig.colorbar(cs_pred, ax=axes[1], shrink=0.7)

        err0 = np.abs(gt_series[0] - pred_series[0])
        err_vmax = float(np.abs(gt_series - pred_series).max())
        cs_err = _tricontour_axis(axes[2], x, y, err0, 0.0, err_vmax, CMAP_ERROR,
                                  "|error|")
        fig.colorbar(cs_err, ax=axes[2], shrink=0.7)

    title = fig.suptitle(f"t = {t_values[0]:.2f}", fontsize=14)

    def update(frame: int):
        nonlocal cs_gt
        # Clear and redraw (tricontourf doesn't support in-place update)
        axes[0].clear()
        cs_gt = _tricontour_axis(axes[0], x, y, gt_series[frame], vmin, vmax, cmap,
                                 f"Ground truth — {field_name}")

        if pred_given:
            axes[1].clear()
            _tricontour_axis(axes[1], x, y, pred_series[frame], vmin, vmax, cmap,
                             f"PINN — {field_name}")
            axes[2].clear()
            err = np.abs(gt_series[frame] - pred_series[frame])
            _tricontour_axis(axes[2], x, y, err, 0.0, err_vmax, CMAP_ERROR, "|error|")

        title.set_text(f"t = {t_values[frame]:.2f}")
        return []

    anim = manim.FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    _save_animation(anim, out_path, fps=fps)
    plt.close(fig)


# -----------------------------------------------------------------------------
# A3: Sparse reconstruction
# -----------------------------------------------------------------------------

def animate_sparse_reconstruction(
    x_obs: np.ndarray,       # (n_obs,) — spatial x of observation points (possibly time-varying)
    y_obs: np.ndarray,       # (n_obs,)
    obs_series: np.ndarray,  # (T, n_obs) — observed u values over time
    x_grid: np.ndarray,      # (N,) — full grid x
    y_grid: np.ndarray,      # (N,)
    pred_series: np.ndarray, # (T, N) — PINN-reconstructed full field
    t_values: np.ndarray,    # (T,)
    out_path: Path,
    *,
    field_name: str = "u",
    cmap: str = CMAP_VORTICITY,
    fps: int = 20,
) -> None:
    """
    Left panel: scatter of sparse observations.
    Right panel: PINN-reconstructed full field.

    Demonstrates PINN's ability to infer full field from sparse data.
    """
    T = pred_series.shape[0]
    vmax = float(np.abs(pred_series).max())
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # Left: scatter observations
    sc = axes[0].scatter(x_obs, y_obs, c=obs_series[0], cmap=cmap, vmin=vmin, vmax=vmax, s=15)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Sparse observations (N={len(x_obs)})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(sc, ax=axes[0], shrink=0.7)

    # Right: reconstructed full field
    cs = _tricontour_axis(axes[1], x_grid, y_grid, pred_series[0], vmin, vmax, cmap,
                          f"PINN reconstruction — {field_name}")
    fig.colorbar(cs, ax=axes[1], shrink=0.7)

    title = fig.suptitle(f"t = {t_values[0]:.2f}", fontsize=14)

    def update(frame: int):
        sc.set_array(obs_series[frame])
        axes[1].clear()
        _tricontour_axis(axes[1], x_grid, y_grid, pred_series[frame], vmin, vmax, cmap,
                         f"PINN reconstruction — {field_name}")
        title.set_text(f"t = {t_values[frame]:.2f}")
        return []

    anim = manim.FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)
    _save_animation(anim, out_path, fps=fps)
    plt.close(fig)


# -----------------------------------------------------------------------------
# A2: Training evolution
# -----------------------------------------------------------------------------

def animate_training_evolution(
    x: np.ndarray,
    y: np.ndarray,
    snapshot_series: np.ndarray,  # (n_snapshots, N) — PINN field at each saved step
    step_values: Sequence[int],    # training step at each snapshot
    out_path: Path,
    *,
    field_name: str = "u",
    cmap: str = CMAP_VORTICITY,
    fps: int = 10,
) -> None:
    """
    Show how the PINN-predicted field evolves during training.

    snapshot_series[k] is the prediction after step_values[k] training steps.
    """
    n_snap = snapshot_series.shape[0]
    vmax = float(np.abs(snapshot_series).max())
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    cs = _tricontour_axis(ax, x, y, snapshot_series[0], vmin, vmax, cmap, field_name)
    fig.colorbar(cs, ax=ax, shrink=0.7)
    title = ax.set_title(f"{field_name} — step {step_values[0]}")

    def update(frame: int):
        ax.clear()
        _tricontour_axis(ax, x, y, snapshot_series[frame], vmin, vmax, cmap, field_name)
        ax.set_title(f"{field_name} — step {step_values[frame]}")
        return []

    anim = manim.FuncAnimation(fig, update, frames=n_snap, interval=1000 // fps, blit=False)
    _save_animation(anim, out_path, fps=fps)
    plt.close(fig)
