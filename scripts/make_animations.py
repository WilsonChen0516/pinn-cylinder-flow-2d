"""
Generate the three animations from trained models.

A1 — Ground truth vs PINN evolution (uses E1 checkpoint)
A2 — Training evolution (uses E1 snapshots saved during training)
A3 — Sparse reconstruction (uses E2 checkpoint)

Usage:
    python scripts/make_animations.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.loader import CylinderWakeDataset
from src.models.pinn import PINN
from src.physics.navier_stokes import velocity_from_streamfunction
from src.visualization.animations import (
    animate_field_evolution,
    animate_sparse_reconstruction,
    animate_training_evolution,
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
        print(f"  no final checkpoint in {ckpt_dir}")
        return None
    ckpt_path = final_candidates[-1]
    print(f"  loading: {ckpt_path}")

    lb, ub = ds.domain.as_lower_upper()
    # For inverse, we need learn_lambdas=True; for forward, False
    # Detect from presence of lambda in checkpoint
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


def _pinn_field_series(model, ds: CylinderWakeDataset, device: str,
                      field: str = "u") -> np.ndarray:
    """Run PINN over all (N*T) ground-truth points, return (T, N)."""
    x = ds.X_star[:, 0]
    y = ds.X_star[:, 1]
    N = ds.N
    T = ds.T

    out = np.zeros((T, N), dtype=np.float32)

    for k in range(T):
        xk = torch.from_numpy(x.reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
        yk = torch.from_numpy(y.reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
        tk = torch.full((N, 1), float(ds.t_star[k]), dtype=torch.float32,
                        device=device).requires_grad_(True)

        psi, _ = model(xk, yk, tk)
        if field == "u":
            u_pred = velocity_from_streamfunction(psi, xk, yk)[0]
            out[k] = u_pred.detach().cpu().numpy().flatten()
        elif field == "v":
            _, v_pred = velocity_from_streamfunction(psi, xk, yk)
            out[k] = v_pred.detach().cpu().numpy().flatten()

    return out


# -----------------------------------------------------------------------------

def make_a1(ds: CylinderWakeDataset, device: str):
    print("[A1] Ground truth vs PINN evolution...")
    model = _load_pinn_from_run("e1_forward", ds, device)
    if model is None:
        print("  SKIP: E1 not trained")
        return

    gt_u = ds.U_star[:, 0, :].T   # (T, N)
    pred_u = _pinn_field_series(model, ds, device, field="u")

    animate_field_evolution(
        x=ds.X_star[:, 0], y=ds.X_star[:, 1],
        gt_series=gt_u, pred_series=pred_u, t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_ground_truth_vs_pinn.mp4",
        field_name="u", fps=20,
    )


def make_a2():
    print("[A2] Training evolution...")
    snap_path = RESULTS_DIR / "e1_forward" / "snapshots" / "snapshots.npz"
    if not snap_path.exists():
        print(f"  SKIP: {snap_path} not found (was snapshot_for_animation=true?)")
        return
    data = np.load(snap_path)
    animate_training_evolution(
        x=data["grid_x"], y=data["grid_y"],
        snapshot_series=data["u"],
        step_values=data["steps"].tolist(),
        out_path=FIGURES_DIR / "anim_training_evolution.mp4",
        field_name="u", fps=8,
    )


def make_a3(ds: CylinderWakeDataset, device: str):
    print("[A3] Sparse observation reconstruction...")
    model = _load_pinn_from_run("e2_inverse_N5000", ds, device)
    if model is None:
        print("  SKIP: E2 not trained")
        return

    # Re-generate same obs points by using same seed
    obs = ds.random_subsample(5000, seed=42)
    x_obs = obs["x"].flatten()
    y_obs = obs["y"].flatten()

    # For the animation, show obs values and PINN field at the same time instants
    T = ds.T
    pred_u = _pinn_field_series(model, ds, device, field="u")

    # Observation values per time step: filter obs indices that happen to be at that time
    # Simpler: for each time step, just show the obs points that fall near that time.
    # We do a nearest-time assignment.
    obs_t = obs["t"].flatten()
    obs_u = obs["u"].flatten()

    obs_series = np.zeros((T, len(x_obs)), dtype=np.float32)
    # Fill each frame by bucketing obs to the nearest time step
    t_vals = ds.t_star
    obs_time_idx = np.array([np.argmin(np.abs(t_vals - ot)) for ot in obs_t])

    # For visualization, show *all* obs scattered in space with static u values
    # (since each obs has a single (x,y,t) -> a single u).
    # We color each obs by its actual u value; per-frame we fade points not at that time.
    # Simpler approach: just show the fixed scatter with static colors, full PINN field changes.
    # Here we use dynamic: obs_series[t, i] = obs_u[i] if its time matches, else np.nan
    for t_idx in range(T):
        mask = obs_time_idx == t_idx
        frame = np.full(len(x_obs), np.nan, dtype=np.float32)
        frame[mask] = obs_u[mask]
        obs_series[t_idx] = frame

    # NaN is a problem for scatter colormap; replace with 0 and set alpha? Keep simple: use obs_u static.
    obs_series_static = np.tile(obs_u[None, :], (T, 1))

    animate_sparse_reconstruction(
        x_obs=x_obs, y_obs=y_obs, obs_series=obs_series_static,
        x_grid=ds.X_star[:, 0], y_grid=ds.X_star[:, 1],
        pred_series=pred_u, t_values=ds.t_star,
        out_path=FIGURES_DIR / "anim_sparse_reconstruction.mp4",
        field_name="u", fps=20,
    )


# -----------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds = CylinderWakeDataset(DATA_PATH)

    make_a1(ds, device)
    make_a2()
    make_a3(ds, device)

    print()
    print(f"Animations written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
