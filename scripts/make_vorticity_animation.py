"""
Generate ground truth vs PINN vorticity comparison animation.

This is the most visually striking animation — vorticity reveals the
Karman vortex street clearly, and side-by-side comparison shows whether
PINN captured the physics correctly.

Output: figures/anim_vorticity_gt_vs_pinn.gif (or .mp4 if ffmpeg available)

Usage:
    python scripts/make_vorticity_animation.py
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

DATA_PATH = PROJECT_ROOT / "data" / "cylinder_nektar_wake.mat"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Which run to use for PINN — E1 forward by default (best forward result)
PINN_RUN = "e1_forward"


# -----------------------------------------------------------------------------

def load_pinn(run_name: str, ds: CylinderWakeDataset, device: str):
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


# -----------------------------------------------------------------------------

def compute_vorticity_pinn(model, ds: CylinderWakeDataset, device: str) -> np.ndarray:
    """
    PINN vorticity = dv/dx - du/dy, computed via autograd
    (much more accurate than finite differences).
    Returns (T, N).
    """
    x = ds.X_star[:, 0]
    y = ds.X_star[:, 1]
    N = ds.N
    T = ds.T

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

        # Autograd-based derivatives — exact (up to fp precision)
        dv_dx = grad(v_pred, xk)
        du_dy = grad(u_pred, yk)
        omega_k = dv_dx - du_dy

        omega[k] = omega_k.detach().cpu().numpy().flatten()

    return omega


def compute_vorticity_gt(ds: CylinderWakeDataset) -> np.ndarray:
    """
    Ground truth vorticity via grid interpolation + finite differences.
    Returns (T, N).
    Falls back to cached version if available to avoid recomputing.
    """
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


# -----------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="e2_inverse_N5000",
                        help="Run name (default: e2_inverse_N5000)")
    parser.add_argument("--out", default=None,
                        help="Output filename. Default: anim_vorticity_<run>.mp4")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Using run: {args.run}")

    ds = CylinderWakeDataset(DATA_PATH)

    print(f"\n[GT vorticity]")
    gt_omega = compute_vorticity_gt(ds)

    print(f"\n[PINN vorticity from run: {args.run}]")
    model = load_pinn(args.run, ds, device)
    pinn_omega = compute_vorticity_pinn(model, ds, device)

    # Quick sanity: max abs values
    print(f"\n  GT  vorticity range:  [{gt_omega.min():+.3f}, {gt_omega.max():+.3f}]")
    print(f"  PINN vorticity range: [{pinn_omega.min():+.3f}, {pinn_omega.max():+.3f}]")

    out_name = args.out or f"anim_vorticity_{args.run}.mp4"
    out_path = FIGURES_DIR / out_name

    print(f"\n[Animation] Rendering to {out_path}...")
    animate_field_evolution(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_series=gt_omega,
        pred_series=pinn_omega,
        t_values=ds.t_star,
        out_path=out_path,
        field_name="vorticity",
        fps=20,
        symmetric_colorbar=True,
    )

    print(f"\nDone. Output saved to {out_path}")


if __name__ == "__main__":
    main()