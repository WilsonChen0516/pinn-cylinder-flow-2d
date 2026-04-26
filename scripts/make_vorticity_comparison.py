"""
Generate E1 vs E2 vorticity comparison animation with UNIFIED colorbar.

Layout (2 rows × 3 cols):
    Row 1 (E1): GT vorticity | E1 PINN vorticity | |error|
    Row 2 (E2): GT vorticity | E2 PINN vorticity | |error|

All vorticity colorbars share the same range (symmetric ±vmax).
All error colorbars share the same range (E1 max as reference, since E1 errs more).

Output: figures/anim_vorticity_e1_vs_e2.mp4

Usage:
    python scripts/make_vorticity_comparison.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from scipy.interpolate import griddata

from src.data.loader import CylinderWakeDataset
from src.models.pinn import PINN
from src.physics.derivatives import grad
from src.physics.navier_stokes import velocity_from_streamfunction
from src.visualization.styles import apply_style, CMAP_VELOCITY, CMAP_ERROR

apply_style()

DATA_PATH = PROJECT_ROOT / "data" / "cylinder_nektar_wake.mat"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

E1_RUN = "e1_ablation_A_baseline"
E2_RUN = "e2_inverse_N5000"


def _save_animation(anim, out_path, fps=20):
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
    writer = manim.PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    print(f"  saved: {out_path}")


def load_pinn_from_run(run_name, ds, device):
    ckpt_dir = RESULTS_DIR / run_name / "checkpoints"
    final = sorted(ckpt_dir.glob("*_final.pt"))
    if not final:
        candidates = sorted(ckpt_dir.glob("step_*.pt"))
        ckpt_path = candidates[-1]
    else:
        ckpt_path = final[-1]
    print(f"  loading {run_name}: {ckpt_path.name}")

    lb, ub = ds.domain.as_lower_upper()
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    has_lambdas = "lambda_1" in payload

    model = PINN(lb=lb, ub=ub, hidden_layers=8, neurons_per_layer=20,
                 activation="tanh", learn_lambdas=has_lambdas).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def compute_pinn_vorticity(model, ds, device):
    """Compute PINN vorticity at all (N, T) using autograd."""
    x = ds.X_star[:, 0]
    y = ds.X_star[:, 1]
    N, T = ds.N, ds.T

    omega = np.zeros((T, N), dtype=np.float32)
    print(f"  computing PINN vorticity at {N * T} points...")
    for k in range(T):
        if k % 50 == 0:
            print(f"    frame {k}/{T}")
        xk = torch.from_numpy(x.reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
        yk = torch.from_numpy(y.reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
        tk = torch.full((N, 1), float(ds.t_star[k]), dtype=torch.float32,
                        device=device).requires_grad_(True)
        psi, _ = model(xk, yk, tk)
        u_pred, v_pred = velocity_from_streamfunction(psi, xk, yk)
        dv_dx = grad(v_pred, xk)
        du_dy = grad(u_pred, yk)
        omega_k = dv_dx - du_dy
        omega[k] = omega_k.detach().cpu().numpy().flatten()
    return omega


def compute_gt_vorticity(ds):
    """Cached ground truth vorticity via finite differences."""
    cache_path = FIGURES_DIR / "vorticity_gt.npy"
    if cache_path.exists():
        print(f"  using cached: {cache_path}")
        return np.load(cache_path)

    x = ds.X_star[:, 0]
    y = ds.X_star[:, 1]
    nx, ny = 200, 100
    xs = np.linspace(x.min(), x.max(), nx)
    ys = np.linspace(y.min(), y.max(), ny)
    XX, YY = np.meshgrid(xs, ys)

    vorticity = np.zeros((ds.T, ds.N), dtype=np.float32)
    print(f"  computing GT vorticity for {ds.T} frames...")
    for k in range(ds.T):
        if k % 50 == 0:
            print(f"    frame {k}/{ds.T}")
        u_field = ds.U_star[:, 0, k]
        v_field = ds.U_star[:, 1, k]
        U = griddata((x, y), u_field, (XX, YY), method="linear")
        V = griddata((x, y), v_field, (XX, YY), method="linear")
        U = np.nan_to_num(U, nan=0.0)
        V = np.nan_to_num(V, nan=0.0)
        dVdx = np.gradient(V, xs, axis=1)
        dUdy = np.gradient(U, ys, axis=0)
        omega_grid = dVdx - dUdy
        omega_pts = griddata((XX.flatten(), YY.flatten()), omega_grid.flatten(),
                             (x, y), method="linear", fill_value=0.0)
        vorticity[k] = omega_pts

    np.save(cache_path, vorticity)
    return vorticity


def animate_2x3_unified(x, y, gt_omega, e1_omega, e2_omega, t_values, out_path, fps=20):
    """
    2 rows × 3 cols layout with UNIFIED colorbar:
        Row 1 (E1): GT | E1 PINN | |error|
        Row 2 (E2): GT | E2 PINN | |error|
    """
    T = gt_omega.shape[0]

    # Unified vorticity range across both PINN and GT
    omega_vmax = float(max(np.abs(gt_omega).max(),
                           np.abs(e1_omega).max(),
                           np.abs(e2_omega).max()))

    # Unified error range — use E1's max (since E1 errs more)
    e1_err_vmax = float(np.abs(gt_omega - e1_omega).max())
    e2_err_vmax = float(np.abs(gt_omega - e2_omega).max())
    err_vmax = e1_err_vmax  # E1's max as the unified reference

    print(f"  unified vorticity colorbar: ±{omega_vmax:.3f}")
    print(f"  unified error colorbar: [0, {err_vmax:.4f}]  (E1 max as reference)")
    print(f"  E2 vorticity error max: {e2_err_vmax:.4f}  ({100*e2_err_vmax/err_vmax:.1f}% of E1)")

    omega_levels = np.linspace(-omega_vmax, omega_vmax, 30)
    err_levels = np.linspace(0, err_vmax, 30)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    plt.subplots_adjust(
        left=0.05, right=0.95,
        top=0.92, bottom=0.06,
        wspace=0.25, hspace=0.30,
    )

    def _draw(ax, field, levels, vmin, vmax, cmap, title):
        cs = ax.tricontourf(x, y, field, levels=levels, cmap=cmap,
                             vmin=vmin, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        return cs

    # --- Static initial frame to set up colorbars ---
    # Row 1: E1
    cs00 = _draw(axes[0, 0], gt_omega[0], omega_levels, -omega_vmax, omega_vmax,
                 CMAP_VELOCITY, "Ground truth — vorticity")
    cs01 = _draw(axes[0, 1], e1_omega[0], omega_levels, -omega_vmax, omega_vmax,
                 CMAP_VELOCITY, "Forward PINN — vorticity")
    cs02 = _draw(axes[0, 2], np.abs(gt_omega[0] - e1_omega[0]),
                 err_levels, 0, err_vmax, CMAP_ERROR, "|error|")
    fig.colorbar(cs00, ax=axes[0, 0], shrink=0.6, aspect=20)
    fig.colorbar(cs01, ax=axes[0, 1], shrink=0.6, aspect=20)
    fig.colorbar(cs02, ax=axes[0, 2], shrink=0.6, aspect=20)

    # Row 2: E2 (same colorbar scales as E1)
    cs10 = _draw(axes[1, 0], gt_omega[0], omega_levels, -omega_vmax, omega_vmax,
                 CMAP_VELOCITY, "Ground truth — vorticity")
    cs11 = _draw(axes[1, 1], e2_omega[0], omega_levels, -omega_vmax, omega_vmax,
                 CMAP_VELOCITY, "Inverse PINN — vorticity")
    cs12 = _draw(axes[1, 2], np.abs(gt_omega[0] - e2_omega[0]),
                 err_levels, 0, err_vmax, CMAP_ERROR, "|error|")
    fig.colorbar(cs10, ax=axes[1, 0], shrink=0.6, aspect=20)
    fig.colorbar(cs11, ax=axes[1, 1], shrink=0.6, aspect=20)
    fig.colorbar(cs12, ax=axes[1, 2], shrink=0.6, aspect=20)

    # # Row group labels
    # fig.text(0.015, 0.74, "E1 Forward",
    #          fontsize=12, fontweight="bold", rotation=90,
    #          va="center", ha="center", color="#1f77b4")
    # fig.text(0.015, 0.27, "E2 Inverse",
    #          fontsize=12, fontweight="bold", rotation=90,
    #          va="center", ha="center", color="#d62728")

    title = fig.suptitle(f"Vorticity Comparison: Forward vs. Inverse (t = {t_values[0]:.2f})",
                          fontsize=13, y=0.98)

    def update(frame):
        # Row 1: E1
        axes[0, 0].clear()
        _draw(axes[0, 0], gt_omega[frame], omega_levels, -omega_vmax, omega_vmax,
              CMAP_VELOCITY, "Ground truth — vorticity")
        axes[0, 1].clear()
        _draw(axes[0, 1], e1_omega[frame], omega_levels, -omega_vmax, omega_vmax,
              CMAP_VELOCITY, "Forward PINN — vorticity")
        axes[0, 2].clear()
        _draw(axes[0, 2], np.abs(gt_omega[frame] - e1_omega[frame]),
              err_levels, 0, err_vmax, CMAP_ERROR, "|error|")

        # Row 2: E2
        axes[1, 0].clear()
        _draw(axes[1, 0], gt_omega[frame], omega_levels, -omega_vmax, omega_vmax,
              CMAP_VELOCITY, "Ground truth — vorticity")
        axes[1, 1].clear()
        _draw(axes[1, 1], e2_omega[frame], omega_levels, -omega_vmax, omega_vmax,
              CMAP_VELOCITY, "Inverse PINN — vorticity")
        axes[1, 2].clear()
        _draw(axes[1, 2], np.abs(gt_omega[frame] - e2_omega[frame]),
              err_levels, 0, err_vmax, CMAP_ERROR, "|error|")

        title.set_text(f"Vorticity Comparison: Forward vs. Inverse (t = {t_values[frame]:.2f})")
        return []

    anim = manim.FuncAnimation(fig, update, frames=T,
                                interval=1000 // fps, blit=False)
    _save_animation(anim, out_path, fps=fps)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds = CylinderWakeDataset(DATA_PATH)

    print(f"\n[GT vorticity]")
    gt_omega = compute_gt_vorticity(ds)

    print(f"\n[E1 PINN vorticity from {E1_RUN}]")
    e1_model = load_pinn_from_run(E1_RUN, ds, device)
    e1_omega = compute_pinn_vorticity(e1_model, ds, device)

    print(f"\n[E2 PINN vorticity from {E2_RUN}]")
    e2_model = load_pinn_from_run(E2_RUN, ds, device)
    e2_omega = compute_pinn_vorticity(e2_model, ds, device)

    print(f"\nRendering 2x3 unified-colorbar comparison animation...")
    animate_2x3_unified(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_omega=gt_omega,
        e1_omega=e1_omega,
        e2_omega=e2_omega,
        t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_vorticity_e1_vs_e2.mp4",
        fps=20,
    )

    print(f"\nDone.")


if __name__ == "__main__":
    main()