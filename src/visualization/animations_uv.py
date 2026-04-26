"""
Two-field animation utilities (u + v, side by side and stacked).

Produces 2-row layouts comparing ground truth vs. PINN reconstruction
for both velocity components simultaneously.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manim

from src.visualization.styles import apply_style, CMAP_VELOCITY, CMAP_ERROR

apply_style()


def _save_animation(anim: manim.FuncAnimation, out_path: Path, fps: int = 20) -> None:
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


def _tricontour(ax, x, y, field, vmin, vmax, cmap, title):
    cs = ax.tricontourf(x, y, field, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return cs


# -----------------------------------------------------------------------------
# A1-uvp: Ground truth vs PINN evolution, u + v + p stacked (3×3)
# -----------------------------------------------------------------------------

def animate_field_evolution_uvp(
    x: np.ndarray,
    y: np.ndarray,
    gt_u: np.ndarray, pred_u: np.ndarray,
    gt_v: np.ndarray, pred_v: np.ndarray,
    gt_p: np.ndarray, pred_p: np.ndarray,
    t_values: np.ndarray,
    out_path: Path,
    *,
    fps: int = 20,
) -> None:
    """
    3 rows × 3 cols animation:
        Row 1: u (GT / PINN / |error|)
        Row 2: v (GT / PINN / |error|)
        Row 3: p (GT / PINN / |error|)  -- p is centered (gauge-invariant)
    """
    T = gt_u.shape[0]
    assert gt_u.shape == pred_u.shape == gt_v.shape == pred_v.shape == gt_p.shape == pred_p.shape
    assert len(t_values) == T

    # Center pressure per-frame (remove gauge offset)
    gt_p_centered = gt_p - gt_p.mean(axis=1, keepdims=True)
    pred_p_centered = pred_p - pred_p.mean(axis=1, keepdims=True)

    # Color limits
    u_vmax = float(max(np.abs(gt_u).max(), np.abs(pred_u).max()))
    v_vmax = float(max(np.abs(gt_v).max(), np.abs(pred_v).max()))
    p_min = float(min(gt_p_centered.min(), pred_p_centered.min()))
    p_max = float(max(gt_p_centered.max(), pred_p_centered.max()))
    u_err_vmax = float(np.abs(gt_u - pred_u).max())
    v_err_vmax = float(np.abs(gt_v - pred_v).max())
    p_err_vmax = float(np.abs(gt_p_centered - pred_p_centered).max())

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), constrained_layout=True)

    # Row 1: u
    cs00 = _tricontour(axes[0, 0], x, y, gt_u[0], -u_vmax, u_vmax, CMAP_VELOCITY,
                       "Ground truth — u")
    cs01 = _tricontour(axes[0, 1], x, y, pred_u[0], -u_vmax, u_vmax, CMAP_VELOCITY,
                       "PINN — u")
    cs02 = _tricontour(axes[0, 2], x, y, np.abs(gt_u[0] - pred_u[0]),
                       0, u_err_vmax, CMAP_ERROR, "|error| — u")
    fig.colorbar(cs00, ax=axes[0, 0], shrink=0.7)
    fig.colorbar(cs01, ax=axes[0, 1], shrink=0.7)
    fig.colorbar(cs02, ax=axes[0, 2], shrink=0.7)

    # Row 2: v
    cs10 = _tricontour(axes[1, 0], x, y, gt_v[0], -v_vmax, v_vmax, CMAP_VELOCITY,
                       "Ground truth — v")
    cs11 = _tricontour(axes[1, 1], x, y, pred_v[0], -v_vmax, v_vmax, CMAP_VELOCITY,
                       "PINN — v")
    cs12 = _tricontour(axes[1, 2], x, y, np.abs(gt_v[0] - pred_v[0]),
                       0, v_err_vmax, CMAP_ERROR, "|error| — v")
    fig.colorbar(cs10, ax=axes[1, 0], shrink=0.7)
    fig.colorbar(cs11, ax=axes[1, 1], shrink=0.7)
    fig.colorbar(cs12, ax=axes[1, 2], shrink=0.7)

    # Row 3: p (centered, gauge-corrected)
    from src.visualization.styles import CMAP_PRESSURE
    cs20 = _tricontour(axes[2, 0], x, y, gt_p_centered[0], p_min, p_max, CMAP_PRESSURE,
                       "Ground truth — p")
    cs21 = _tricontour(axes[2, 1], x, y, pred_p_centered[0], p_min, p_max, CMAP_PRESSURE,
                       "PINN — p  (no observations!)")
    cs22 = _tricontour(axes[2, 2], x, y, np.abs(gt_p_centered[0] - pred_p_centered[0]),
                       0, p_err_vmax, CMAP_ERROR, "|error| — p")
    fig.colorbar(cs20, ax=axes[2, 0], shrink=0.7)
    fig.colorbar(cs21, ax=axes[2, 1], shrink=0.7)
    fig.colorbar(cs22, ax=axes[2, 2], shrink=0.7)

    title = fig.suptitle(f"t = {t_values[0]:.2f}", fontsize=14)

    def update(frame):
        # Row 1: u
        axes[0, 0].clear()
        _tricontour(axes[0, 0], x, y, gt_u[frame], -u_vmax, u_vmax, CMAP_VELOCITY,
                    "Ground truth — u")
        axes[0, 1].clear()
        _tricontour(axes[0, 1], x, y, pred_u[frame], -u_vmax, u_vmax, CMAP_VELOCITY,
                    "PINN — u")
        axes[0, 2].clear()
        _tricontour(axes[0, 2], x, y, np.abs(gt_u[frame] - pred_u[frame]),
                    0, u_err_vmax, CMAP_ERROR, "|error| — u")

        # Row 2: v
        axes[1, 0].clear()
        _tricontour(axes[1, 0], x, y, gt_v[frame], -v_vmax, v_vmax, CMAP_VELOCITY,
                    "Ground truth — v")
        axes[1, 1].clear()
        _tricontour(axes[1, 1], x, y, pred_v[frame], -v_vmax, v_vmax, CMAP_VELOCITY,
                    "PINN — v")
        axes[1, 2].clear()
        _tricontour(axes[1, 2], x, y, np.abs(gt_v[frame] - pred_v[frame]),
                    0, v_err_vmax, CMAP_ERROR, "|error| — v")

        # Row 3: p (centered)
        axes[2, 0].clear()
        _tricontour(axes[2, 0], x, y, gt_p_centered[frame], p_min, p_max, CMAP_PRESSURE,
                    "Ground truth — p")
        axes[2, 1].clear()
        _tricontour(axes[2, 1], x, y, pred_p_centered[frame], p_min, p_max, CMAP_PRESSURE,
                    "PINN — p  (no observations!)")
        axes[2, 2].clear()
        _tricontour(axes[2, 2], x, y, np.abs(gt_p_centered[frame] - pred_p_centered[frame]),
                    0, p_err_vmax, CMAP_ERROR, "|error| — p")

        title.set_text(f"t = {t_values[frame]:.2f}")
        return []

    anim = manim.FuncAnimation(fig, update, frames=T,
                               interval=1000 // fps, blit=False)
    _save_animation(anim, out_path, fps=fps)
    plt.close(fig)


# -----------------------------------------------------------------------------
# A1-full: Ground truth vs PINN evolution, BOTH u and v stacked
# -----------------------------------------------------------------------------

def animate_field_evolution_uv(
    x: np.ndarray,
    y: np.ndarray,
    gt_u: np.ndarray, pred_u: np.ndarray,    # (T, N)
    gt_v: np.ndarray, pred_v: np.ndarray,    # (T, N)
    t_values: np.ndarray,
    out_path: Path,
    *,
    fps: int = 20,
) -> None:
    """
    2 rows × 3 cols animation:
        Row 1: u (GT / PINN / |error|)
        Row 2: v (GT / PINN / |error|)
    """
    T = gt_u.shape[0]
    assert gt_u.shape == pred_u.shape == gt_v.shape == pred_v.shape
    assert len(t_values) == T

    # Symmetric color limits for each component
    u_vmax = float(max(np.abs(gt_u).max(), np.abs(pred_u).max()))
    v_vmax = float(max(np.abs(gt_v).max(), np.abs(pred_v).max()))
    u_err_vmax = float(np.abs(gt_u - pred_u).max())
    v_err_vmax = float(np.abs(gt_v - pred_v).max())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    # Row 1: u
    cs00 = _tricontour(axes[0, 0], x, y, gt_u[0], -u_vmax, u_vmax, CMAP_VELOCITY,
                       "Ground truth — u")
    cs01 = _tricontour(axes[0, 1], x, y, pred_u[0], -u_vmax, u_vmax, CMAP_VELOCITY,
                       "PINN — u")
    cs02 = _tricontour(axes[0, 2], x, y, np.abs(gt_u[0] - pred_u[0]),
                       0, u_err_vmax, CMAP_ERROR, "|error| — u")
    fig.colorbar(cs00, ax=axes[0, 0], shrink=0.7)
    fig.colorbar(cs01, ax=axes[0, 1], shrink=0.7)
    fig.colorbar(cs02, ax=axes[0, 2], shrink=0.7)

    # Row 2: v
    cs10 = _tricontour(axes[1, 0], x, y, gt_v[0], -v_vmax, v_vmax, CMAP_VELOCITY,
                       "Ground truth — v")
    cs11 = _tricontour(axes[1, 1], x, y, pred_v[0], -v_vmax, v_vmax, CMAP_VELOCITY,
                       "PINN — v")
    cs12 = _tricontour(axes[1, 2], x, y, np.abs(gt_v[0] - pred_v[0]),
                       0, v_err_vmax, CMAP_ERROR, "|error| — v")
    fig.colorbar(cs10, ax=axes[1, 0], shrink=0.7)
    fig.colorbar(cs11, ax=axes[1, 1], shrink=0.7)
    fig.colorbar(cs12, ax=axes[1, 2], shrink=0.7)

    title = fig.suptitle(f"t = {t_values[0]:.2f}", fontsize=14)

    def update(frame):
        # Row 1: u
        axes[0, 0].clear()
        _tricontour(axes[0, 0], x, y, gt_u[frame], -u_vmax, u_vmax, CMAP_VELOCITY,
                    "Ground truth — u")
        axes[0, 1].clear()
        _tricontour(axes[0, 1], x, y, pred_u[frame], -u_vmax, u_vmax, CMAP_VELOCITY,
                    "PINN — u")
        axes[0, 2].clear()
        _tricontour(axes[0, 2], x, y, np.abs(gt_u[frame] - pred_u[frame]),
                    0, u_err_vmax, CMAP_ERROR, "|error| — u")

        # Row 2: v
        axes[1, 0].clear()
        _tricontour(axes[1, 0], x, y, gt_v[frame], -v_vmax, v_vmax, CMAP_VELOCITY,
                    "Ground truth — v")
        axes[1, 1].clear()
        _tricontour(axes[1, 1], x, y, pred_v[frame], -v_vmax, v_vmax, CMAP_VELOCITY,
                    "PINN — v")
        axes[1, 2].clear()
        _tricontour(axes[1, 2], x, y, np.abs(gt_v[frame] - pred_v[frame]),
                    0, v_err_vmax, CMAP_ERROR, "|error| — v")

        title.set_text(f"t = {t_values[frame]:.2f}")
        return []

    anim = manim.FuncAnimation(fig, update, frames=T,
                               interval=1000 // fps, blit=False)
    _save_animation(anim, out_path, fps=fps)
    plt.close(fig)


# -----------------------------------------------------------------------------
# A3-full: sparse reconstruction with u and v
# -----------------------------------------------------------------------------

def animate_sparse_reconstruction_uv(
    x_obs: np.ndarray, y_obs: np.ndarray,
    obs_u: np.ndarray,        # (n_obs,) static colors
    obs_v: np.ndarray,
    x_grid: np.ndarray, y_grid: np.ndarray,
    pred_u_series: np.ndarray,   # (T, N)
    pred_v_series: np.ndarray,
    t_values: np.ndarray,
    out_path: Path,
    *,
    fps: int = 20,
) -> None:
    """
    2 rows × 2 cols animation:
        Row 1: scatter obs (u colors) | PINN reconstruction u
        Row 2: scatter obs (v colors) | PINN reconstruction v
    """
    T = pred_u_series.shape[0]

    u_vmax = float(np.abs(pred_u_series).max())
    v_vmax = float(np.abs(pred_v_series).max())

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # Row 1: u
    sc0 = axes[0, 0].scatter(x_obs, y_obs, c=obs_u, cmap=CMAP_VELOCITY,
                              vmin=-u_vmax, vmax=u_vmax, s=12)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title(f"Sparse observations — u (N={len(x_obs)})", fontsize=11)
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")
    fig.colorbar(sc0, ax=axes[0, 0], shrink=0.7)

    cs_u = _tricontour(axes[0, 1], x_grid, y_grid, pred_u_series[0],
                       -u_vmax, u_vmax, CMAP_VELOCITY,
                       "PINN reconstruction — u")
    fig.colorbar(cs_u, ax=axes[0, 1], shrink=0.7)

    # Row 2: v
    sc1 = axes[1, 0].scatter(x_obs, y_obs, c=obs_v, cmap=CMAP_VELOCITY,
                              vmin=-v_vmax, vmax=v_vmax, s=12)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_title(f"Sparse observations — v (N={len(x_obs)})", fontsize=11)
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
    fig.colorbar(sc1, ax=axes[1, 0], shrink=0.7)

    cs_v = _tricontour(axes[1, 1], x_grid, y_grid, pred_v_series[0],
                       -v_vmax, v_vmax, CMAP_VELOCITY,
                       "PINN reconstruction — v")
    fig.colorbar(cs_v, ax=axes[1, 1], shrink=0.7)

    title = fig.suptitle(f"t = {t_values[0]:.2f}", fontsize=14)

    def update(frame):
        axes[0, 1].clear()
        _tricontour(axes[0, 1], x_grid, y_grid, pred_u_series[frame],
                    -u_vmax, u_vmax, CMAP_VELOCITY,
                    "PINN reconstruction — u")
        axes[1, 1].clear()
        _tricontour(axes[1, 1], x_grid, y_grid, pred_v_series[frame],
                    -v_vmax, v_vmax, CMAP_VELOCITY,
                    "PINN reconstruction — v")
        title.set_text(f"t = {t_values[frame]:.2f}")
        return []

    anim = manim.FuncAnimation(fig, update, frames=T,
                               interval=1000 // fps, blit=False)
    _save_animation(anim, out_path, fps=fps)
    plt.close(fig)