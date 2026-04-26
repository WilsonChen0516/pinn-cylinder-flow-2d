"""
Generate ground truth vs PINN comparison animation for E2 (inverse problem).

Outputs both u-only and u+v versions, similar to E1's animations,
but using the E2 trained model.

This complements anim_sparse_reconstruction.gif, providing a direct
GT vs PINN comparison with absolute error maps.

Usage:
    python scripts/make_e2_comparison_animation.py
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
from src.visualization.animations import animate_field_evolution
from src.visualization.animations_uv import (
    animate_field_evolution_uv,
    animate_field_evolution_uvp,
)

DATA_PATH = PROJECT_ROOT / "data" / "cylinder_nektar_wake.mat"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Which E2 run to use. If you have multiple, pick the best one.
E2_RUN = "e2_inverse_N5000"


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

    if has_lambdas:
        print(f"  identified λ₁ = {payload['lambda_1']:.6f}")
        print(f"  identified λ₂ = {payload['lambda_2']:.8f}")

    return model


def pinn_uv_series(model, ds: CylinderWakeDataset, device: str):
    """Run PINN at all (N*T) points; return (T, N) for u, v, and p."""
    x = ds.X_star[:, 0]
    y = ds.X_star[:, 1]
    N = ds.N
    T = ds.T

    u_out = np.zeros((T, N), dtype=np.float32)
    v_out = np.zeros((T, N), dtype=np.float32)
    p_out = np.zeros((T, N), dtype=np.float32)

    print(f"  inference at {N * T} points...")
    for k in range(T):
        if k % 50 == 0:
            print(f"    frame {k}/{T}")
        xk = torch.from_numpy(x.reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
        yk = torch.from_numpy(y.reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
        tk = torch.full((N, 1), float(ds.t_star[k]), dtype=torch.float32,
                        device=device).requires_grad_(True)

        psi, p_pred = model(xk, yk, tk)
        u_pred, v_pred = velocity_from_streamfunction(psi, xk, yk)
        u_out[k] = u_pred.detach().cpu().numpy().flatten()
        v_out[k] = v_pred.detach().cpu().numpy().flatten()
        p_out[k] = p_pred.detach().cpu().numpy().flatten()

    return u_out, v_out, p_out


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Using E2 run: {E2_RUN}")

    ds = CylinderWakeDataset(DATA_PATH)
    model = load_pinn_from_run(E2_RUN, ds, device)

    print("\nRunning inference...")
    pred_u, pred_v, pred_p = pinn_uv_series(model, ds, device)

    gt_u = ds.U_star[:, 0, :].T  # (T, N)
    gt_v = ds.U_star[:, 1, :].T
    gt_p = ds.p_star.T           # (T, N)

    # --- u-only version (matches E1's anim_ground_truth_vs_pinn.mp4 style) ---
    print("\n[E2 u-only comparison]")
    animate_field_evolution(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_series=gt_u, pred_series=pred_u, t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_e2_ground_truth_vs_pinn.mp4",
        field_name="u (E2 inverse)",
        fps=20,
    )

    # --- u+v version ---
    print("\n[E2 u+v comparison]")
    animate_field_evolution_uv(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_u=gt_u, pred_u=pred_u,
        gt_v=gt_v, pred_v=pred_v,
        t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_e2_ground_truth_vs_pinn_uv.mp4",
        fps=20,
    )

    # --- u+v+p version (3x3) ---
    print("\n[E2 u+v+p comparison]")
    animate_field_evolution_uvp(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_u=gt_u, pred_u=pred_u,
        gt_v=gt_v, pred_v=pred_v,
        gt_p=gt_p, pred_p=pred_p,
        t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_e2_ground_truth_vs_pinn_uvp.mp4",
        fps=20,
    )

    print(f"\nDone. Animations saved to: {FIGURES_DIR}")
    print("  anim_e2_ground_truth_vs_pinn.mp4     (3-col: GT / PINN / error, u only)")
    print("  anim_e2_ground_truth_vs_pinn_uv.mp4  (2x3: u and v stacked)")
    print("  anim_e2_ground_truth_vs_pinn_uvp.mp4 (3x3: u, v, p stacked)")


if __name__ == "__main__":
    main()