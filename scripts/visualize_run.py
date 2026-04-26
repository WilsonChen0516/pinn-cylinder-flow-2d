"""
Visualize field predictions from a trained checkpoint.

Loads a model from results/<run_name>/checkpoints/*_final.pt, runs inference
on the Raissi ground truth grid, and produces a 3x3 comparison figure
(u, v, p) x (ground truth, PINN, absolute error).

Usage:
    python scripts/visualize_run.py --run e1_forward
    python scripts/visualize_run.py --run e1_forward --time-index 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.loader import CylinderWakeDataset
from src.models.pinn import PINN
from src.physics.navier_stokes import velocity_from_streamfunction
from src.visualization.plots import plot_field_comparison

DATA_PATH = PROJECT_ROOT / "data" / "cylinder_nektar_wake.mat"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_pinn_checkpoint(run_name: str, ds: CylinderWakeDataset, device: str) -> PINN:
    """Load the final checkpoint of a given run."""
    ckpt_dir = RESULTS_DIR / run_name / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory: {ckpt_dir}")

    # Prefer '*_final.pt' if it exists, else the latest step
    final = sorted(ckpt_dir.glob("*_final.pt"))
    if final:
        ckpt_path = final[-1]
    else:
        ckpts = sorted(ckpt_dir.glob("step_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
        ckpt_path = ckpts[-1]

    print(f"Loading: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    lb, ub = ds.domain.as_lower_upper()
    has_lambdas = "lambda_1" in payload

    model = PINN(
        lb=lb, ub=ub,
        hidden_layers=8, neurons_per_layer=20, activation="tanh",
        learn_lambdas=has_lambdas,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    if has_lambdas:
        print(f"  lambda_1 = {payload['lambda_1']:.6f}  (true 1.0)")
        print(f"  lambda_2 = {payload['lambda_2']:.6f}  (true 0.01)")

    return model


def infer_at_time(
    model: PINN, ds: CylinderWakeDataset, time_index: int, device: str
) -> dict:
    """Run PINN inference at one time step, return (u, v, p) arrays shaped (N,)."""
    snap = ds.snapshot(time_index)
    N = ds.N

    x = torch.from_numpy(snap["x"].reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
    y = torch.from_numpy(snap["y"].reshape(-1, 1).astype(np.float32)).to(device).requires_grad_(True)
    t = torch.full((N, 1), snap["t"], dtype=torch.float32, device=device).requires_grad_(True)

    psi, p_pred = model(x, y, t)
    u_pred, v_pred = velocity_from_streamfunction(psi, x, y)

    return {
        "u": u_pred.detach().cpu().numpy().flatten(),
        "v": v_pred.detach().cpu().numpy().flatten(),
        "p": p_pred.detach().cpu().numpy().flatten(),
        "u_true": snap["u"],
        "v_true": snap["v"],
        "p_true": snap["p"],
        "x": snap["x"],
        "y": snap["y"],
        "t": snap["t"],
    }


def analyze_pressure_scale(p_pred: np.ndarray, p_true: np.ndarray) -> None:
    """Diagnose whether p is 'shape-correct but scale-off' or truly wrong."""
    print("\n--- Pressure diagnostic ---")
    print(f"  p_true   mean={p_true.mean():+.4f}  std={p_true.std():.4f}  "
          f"range=[{p_true.min():+.4f}, {p_true.max():+.4f}]")
    print(f"  p_pred   mean={p_pred.mean():+.4f}  std={p_pred.std():.4f}  "
          f"range=[{p_pred.min():+.4f}, {p_pred.max():+.4f}]")

    # Center both (remove constant offset — pressure only defined up to constant)
    pc_true = p_true - p_true.mean()
    pc_pred = p_pred - p_pred.mean()

    # Best linear fit: p_pred ≈ a * p_true + b
    # With centered: a = <pc_pred, pc_true> / <pc_true, pc_true>
    a = (pc_pred * pc_true).sum() / ((pc_true * pc_true).sum() + 1e-12)
    print(f"  best linear scale factor a = {a:+.4f}")
    print(f"    (if a ≈ 1, p is correctly scaled; if a ≠ 1, p has scale error)")

    # Correlation — does the shape match?
    corr = np.corrcoef(pc_true, pc_pred)[0, 1]
    print(f"  shape correlation = {corr:.4f}")
    print(f"    (>0.9 means shape is correct, just wrong magnitude)")

    # L2 with scale correction
    pc_pred_corrected = pc_pred / (a + 1e-12)
    scaled_l2 = np.linalg.norm(pc_pred_corrected - pc_true) / (np.linalg.norm(pc_true) + 1e-12)
    print(f"  L2 after scale correction = {scaled_l2 * 100:.2f}%")
    print(f"    (if this is small, the PINN learned the right shape)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Run name (e.g., 'e1_forward')")
    parser.add_argument("--time-index", type=int, default=100,
                        help="Time step index to visualize (0-199)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CylinderWakeDataset(DATA_PATH)
    model = load_pinn_checkpoint(args.run, ds, device)

    fields = infer_at_time(model, ds, args.time_index, device)

    # Print per-time-step L2 errors (sanity check vs. global numbers)
    print(f"\n--- Per-time-step L2 errors at t={fields['t']:.2f} ---")
    for name in ("u", "v", "p"):
        pred = fields[name]
        true = fields[f"{name}_true"]
        if name == "p":
            pred = pred - pred.mean()
            true = true - true.mean()
        l2 = np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)
        print(f"  {name}: relative L2 = {l2 * 100:.2f}%")

    # Pressure-specific diagnostic
    analyze_pressure_scale(fields["p"], fields["p_true"])

    # Save the 3x3 comparison figure
    out_dir = RESULTS_DIR / args.run / "figures"
    out_path = out_dir / f"field_comparison_t{args.time_index:03d}.png"
    plot_field_comparison(
        x=fields["x"], y=fields["y"],
        gt={"u": fields["u_true"], "v": fields["v_true"], "p": fields["p_true"]},
        pred={"u": fields["u"], "v": fields["v"], "p": fields["p"]},
        out_path=out_path,
        time_label=f"Run: {args.run}  |  t = {fields['t']:.2f}",
    )

    print(f"\nFigure saved to: {out_path}")
    print("\nOpen the figure and check:")
    print("  1. Does PINN u match GT u?  (should, since L2 < 5%)")
    print("  2. Does PINN v match GT v?  (should mostly match)")
    print("  3. Does PINN p have the same SHAPE as GT p?")
    print("     -> If yes: 'shape correct, scale off', acceptable for E1")
    print("     -> If no:  p is genuinely wrong, needs training adjustment")


if __name__ == "__main__":
    main()