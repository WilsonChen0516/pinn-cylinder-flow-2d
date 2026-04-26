"""
Generate u+v combined animations.

Two animations:
  A1-full: ground truth vs PINN, showing both u AND v stacked (2 rows × 3 cols)
  A3-full: sparse reconstruction, showing both u AND v (2 rows × 2 cols)

Usage:
    python scripts/make_animations_full.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.loader import CylinderWakeDataset
from src.models.pinn import PINN
from src.physics.navier_stokes import velocity_from_streamfunction
from src.visualization.animations_uv import (
    animate_field_evolution_uv,
    animate_sparse_reconstruction_uv,
)

DATA_PATH = PROJECT_ROOT / "data" / "cylinder_nektar_wake.mat"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


# -----------------------------------------------------------------------------

def _load_pinn_from_run(run_name: str, ds: CylinderWakeDataset, device: str):
    """Load PINN model from the 'final' checkpoint of a given run."""
    ckpt_dir = RESULTS_DIR / run_name / "checkpoints"
    final_candidates = sorted(ckpt_dir.glob("*_final.pt"))
    if not final_candidates:
        # Try just any checkpoint named step_*.pt
        candidates = sorted(ckpt_dir.glob("step_*.pt"))
        if not candidates:
            print(f"  no checkpoint found in {ckpt_dir}")
            return None
        ckpt_path = candidates[-1]
    else:
        ckpt_path = final_candidates[-1]
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


def _pinn_uv_series(model, ds: CylinderWakeDataset, device: str):
    """Run PINN at all (N*T) ground truth points; return (T, N) for both u and v."""
    x = ds.X_star[:, 0]
    y = ds.X_star[:, 1]
    N = ds.N
    T = ds.T

    u_out = np.zeros((T, N), dtype=np.float32)
    v_out = np.zeros((T, N), dtype=np.float32)

    print(f"  running PINN inference at {N * T} points...")
    for k in range(T):
        if k % 50 == 0:
            print(f"    frame {k}/{T}")
        xk = torch.from_numpy(x.reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
        yk = torch.from_numpy(y.reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
        tk = torch.full((N, 1), float(ds.t_star[k]), dtype=torch.float32,
                        device=device).requires_grad_(True)

        psi, _ = model(xk, yk, tk)
        u_pred, v_pred = velocity_from_streamfunction(psi, xk, yk)
        u_out[k] = u_pred.detach().cpu().numpy().flatten()
        v_out[k] = v_pred.detach().cpu().numpy().flatten()

    return u_out, v_out


# -----------------------------------------------------------------------------

def make_a1_full(ds: CylinderWakeDataset, device: str):
    print("\n[A1-full] u + v ground truth vs PINN evolution...")
    model = _load_pinn_from_run("e1_forward", ds, device)
    if model is None:
        print("  SKIP: e1_forward not trained")
        return

    pred_u, pred_v = _pinn_uv_series(model, ds, device)
    gt_u = ds.U_star[:, 0, :].T   # (T, N)
    gt_v = ds.U_star[:, 1, :].T

    animate_field_evolution_uv(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_u=gt_u, pred_u=pred_u,
        gt_v=gt_v, pred_v=pred_v,
        t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_ground_truth_vs_pinn_uv.mp4",
        fps=20,
    )


def make_a3_full(ds: CylinderWakeDataset, device: str):
    print("\n[A3-full] u + v sparse reconstruction...")
    # Use the inverse run that has the best results (E2 fix)
    model = _load_pinn_from_run("e2_inverse_N5000_1", ds, device)
    if model is None:
        # Fallback to original E2 run
        model = _load_pinn_from_run("e2_inverse_N5000", ds, device)
    if model is None:
        print("  SKIP: no E2 inverse run found")
        return

    obs = ds.random_subsample(5000, seed=42)
    x_obs = obs["x"].flatten()
    y_obs = obs["y"].flatten()
    obs_u = obs["u"].flatten()
    obs_v = obs["v"].flatten()

    pred_u, pred_v = _pinn_uv_series(model, ds, device)

    animate_sparse_reconstruction_uv(
        x_obs=x_obs, y_obs=y_obs,
        obs_u=obs_u, obs_v=obs_v,
        x_grid=ds.X_star[:, 0], y_grid=ds.X_star[:, 1],
        pred_u_series=pred_u, pred_v_series=pred_v,
        t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_sparse_reconstruction_uv.mp4",
        fps=20,
    )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds = CylinderWakeDataset(DATA_PATH)

    make_a1_full(ds, device)
    make_a3_full(ds, device)

    print(f"\nDone. Animations saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()