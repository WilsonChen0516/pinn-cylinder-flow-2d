"""
Generate E1 forward problem animations from A baseline run:
  - anim_e1_ground_truth_vs_pinn.mp4      (1×3, u only)
  - anim_e1_ground_truth_vs_pinn_uv.mp4   (2×3, u, v)
  - anim_e1_ground_truth_vs_pinn_uvp.mp4  (3×3, u, v, p)
  - anim_e1_vorticity_gt_vs_pinn.mp4      (1×3, vorticity)

Usage:
    python scripts/make_e1_comparison_animation.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy.interpolate import griddata

from src.data.loader import CylinderWakeDataset
from src.models.pinn import PINN
from src.physics.derivatives import grad
from src.physics.navier_stokes import velocity_from_streamfunction
from src.visualization.animations import animate_field_evolution
from src.visualization.animations_uv import (
    animate_field_evolution_uv,
    animate_field_evolution_uvp,
)

DATA_PATH = PROJECT_ROOT / "data" / "cylinder_nektar_wake.mat"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

E1_RUN = "e1_ablation_A_baseline"


def load_pinn_from_run(run_name: str, ds: CylinderWakeDataset, device: str):
    ckpt_dir = RESULTS_DIR / run_name / "checkpoints"
    final = sorted(ckpt_dir.glob("*_final.pt"))
    if not final:
        candidates = sorted(ckpt_dir.glob("step_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
        ckpt_path = candidates[-1]
    else:
        ckpt_path = final[-1]
    print(f"  loading: {ckpt_path}")

    lb, ub = ds.domain.as_lower_upper()
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    has_lambdas = "lambda_1" in payload

    model = PINN(
        lb=lb, ub=ub,
        hidden_layers=8, neurons_per_layer=20, activation="tanh",
        learn_lambdas=has_lambdas,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def pinn_inference_series(model, ds: CylinderWakeDataset, device: str):
    """Run PINN at all (N*T) points; return (T, N) for u, v, p, and vorticity."""
    x = ds.X_star[:, 0]
    y = ds.X_star[:, 1]
    N = ds.N
    T = ds.T

    u_out = np.zeros((T, N), dtype=np.float32)
    v_out = np.zeros((T, N), dtype=np.float32)
    p_out = np.zeros((T, N), dtype=np.float32)
    vort_out = np.zeros((T, N), dtype=np.float32)

    print(f"  inference + vorticity at {N * T} points...")
    for k in range(T):
        if k % 50 == 0:
            print(f"    frame {k}/{T}")
        xk = torch.from_numpy(x.reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
        yk = torch.from_numpy(y.reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
        tk = torch.full((N, 1), float(ds.t_star[k]), dtype=torch.float32,
                        device=device).requires_grad_(True)

        psi, p_pred = model(xk, yk, tk)
        u_pred, v_pred = velocity_from_streamfunction(psi, xk, yk)

        # Compute vorticity using autograd: omega = dv/dx - du/dy
        dv_dx = grad(v_pred, xk)
        du_dy = grad(u_pred, yk)
        omega_k = dv_dx - du_dy

        u_out[k] = u_pred.detach().cpu().numpy().flatten()
        v_out[k] = v_pred.detach().cpu().numpy().flatten()
        p_out[k] = p_pred.detach().cpu().numpy().flatten()
        vort_out[k] = omega_k.detach().cpu().numpy().flatten()

    return u_out, v_out, p_out, vort_out


def compute_vorticity_gt(ds: CylinderWakeDataset) -> np.ndarray:
    """Ground truth vorticity via grid interpolation + finite differences."""
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

        omega_pts = griddata(
            (XX.flatten(), YY.flatten()),
            omega_grid.flatten(),
            (x, y),
            method="linear", fill_value=0.0,
        )
        vorticity[k] = omega_pts

    np.save(cache_path, vorticity)
    print(f"  cached to: {cache_path}")
    return vorticity


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Using E1 run: {E1_RUN}")

    ds = CylinderWakeDataset(DATA_PATH)
    model = load_pinn_from_run(E1_RUN, ds, device)

    print("\nRunning PINN inference (u, v, p, vorticity)...")
    pred_u, pred_v, pred_p, pred_vort = pinn_inference_series(model, ds, device)

    gt_u = ds.U_star[:, 0, :].T   # (T, N)
    gt_v = ds.U_star[:, 1, :].T
    gt_p = ds.p_star.T            # (T, N)

    print("\nComputing GT vorticity...")
    gt_vort = compute_vorticity_gt(ds)

    # --- u-only animation ---
    print("\n[E1 u-only comparison]")
    animate_field_evolution(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_series=gt_u, pred_series=pred_u, t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_e1_ground_truth_vs_pinn.mp4",
        field_name="u",
        fps=20,
    )

    # --- u+v animation ---
    print("\n[E1 u+v comparison]")
    animate_field_evolution_uv(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_u=gt_u, pred_u=pred_u,
        gt_v=gt_v, pred_v=pred_v,
        t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_e1_ground_truth_vs_pinn_uv.mp4",
        fps=20,
    )

    # --- u+v+p animation (3x3) ---
    print("\n[E1 u+v+p comparison]")
    animate_field_evolution_uvp(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_u=gt_u, pred_u=pred_u,
        gt_v=gt_v, pred_v=pred_v,
        gt_p=gt_p, pred_p=pred_p,
        t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_e1_ground_truth_vs_pinn_uvp.mp4",
        fps=20,
    )

    # --- vorticity animation ---
    print("\n[E1 vorticity comparison]")
    print(f"  GT  vorticity range:  [{gt_vort.min():+.3f}, {gt_vort.max():+.3f}]")
    print(f"  PINN vorticity range: [{pred_vort.min():+.3f}, {pred_vort.max():+.3f}]")
    animate_field_evolution(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_series=gt_vort, pred_series=pred_vort, t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_e1_vorticity_gt_vs_pinn.mp4",
        field_name="vorticity",
        fps=20,
        symmetric_colorbar=True,
    )

    print(f"\nDone. Animations saved to: {FIGURES_DIR}")
    print("  anim_e1_ground_truth_vs_pinn.mp4      (1x3: u)")
    print("  anim_e1_ground_truth_vs_pinn_uv.mp4   (2x3: u, v)")
    print("  anim_e1_ground_truth_vs_pinn_uvp.mp4  (3x3: u, v, p)")
    print("  anim_e1_vorticity_gt_vs_pinn.mp4      (1x3: vorticity)")


if __name__ == "__main__":
    main()