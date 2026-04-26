"""
Generate ground truth animation from Raissi data.

This runs without any PINN training — useful as a Day-1 sanity check
and produces the left half of A1 animation.

Usage:
    python scripts/make_ground_truth_animation.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data.loader import CylinderWakeDataset
from src.visualization.animations import animate_field_evolution

DATA_PATH = PROJECT_ROOT / "data" / "cylinder_nektar_wake.mat"
OUT_DIR = PROJECT_ROOT / "figures"


def compute_vorticity_approx(ds: CylinderWakeDataset) -> np.ndarray:
    """
    Approximate vorticity omega = dv/dx - du/dy via scipy griddata + numpy gradient.

    Returns (T, N) array where column ordering matches ds.X_star.

    Note: this is an approximation since ds.X_star may not be on a regular grid.
    For visualization only.
    """
    from scipy.interpolate import griddata

    x = ds.X_star[:, 0]
    y = ds.X_star[:, 1]

    # Build a regular grid for finite differences
    nx, ny = 200, 100
    xs = np.linspace(x.min(), x.max(), nx)
    ys = np.linspace(y.min(), y.max(), ny)
    XX, YY = np.meshgrid(xs, ys)

    vorticity = np.zeros((ds.T, ds.N), dtype=np.float32)

    print(f"  Computing vorticity for {ds.T} frames (this may take a minute)...")
    for k in range(ds.T):
        if k % 20 == 0:
            print(f"    frame {k}/{ds.T}")
        u_field = ds.U_star[:, 0, k]
        v_field = ds.U_star[:, 1, k]

        # Interpolate to regular grid
        U = griddata((x, y), u_field, (XX, YY), method="linear")
        V = griddata((x, y), v_field, (XX, YY), method="linear")

        # Replace NaN (outside convex hull) with 0 for gradient computation
        U = np.nan_to_num(U, nan=0.0)
        V = np.nan_to_num(V, nan=0.0)

        # omega = dv/dx - du/dy
        dVdx = np.gradient(V, xs, axis=1)
        dUdy = np.gradient(U, ys, axis=0)
        omega_grid = dVdx - dUdy  # (ny, nx)

        # Interpolate back to original scatter points
        omega_pts = griddata(
            (XX.flatten(), YY.flatten()),
            omega_grid.flatten(),
            (x, y),
            method="linear",
            fill_value=0.0,
        )
        vorticity[k] = omega_pts

    return vorticity


def main() -> None:
    print("=" * 60)
    print("Ground Truth Animation Generation")
    print("=" * 60)

    ds = CylinderWakeDataset(DATA_PATH)
    print(ds.summary())
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x = ds.X_star[:, 0]
    y = ds.X_star[:, 1]

    # --- Animation 1: u velocity field (fast, no vorticity) ---
    print("Generating u-velocity animation...")
    u_series = ds.U_star[:, 0, :].T  # (T, N)
    animate_field_evolution(
        x, y,
        gt_series=u_series,
        pred_series=None,  # GT only for now
        t_values=ds.t_star,
        out_path=OUT_DIR / "gt_u_velocity.mp4",
        field_name="u velocity",
        fps=20,
    )

    # --- Animation 2: vorticity (best visualization of Karman street) ---
    print("\nGenerating vorticity animation (Karman vortex street)...")
    print("  [Note: this computes vorticity via grid interpolation + finite diff]")
    vorticity = compute_vorticity_approx(ds)
    animate_field_evolution(
        x, y,
        gt_series=vorticity,
        pred_series=None,
        t_values=ds.t_star,
        out_path=OUT_DIR / "gt_vorticity.mp4",
        field_name="vorticity",
        fps=20,
    )

    # Save vorticity for later reuse (don't recompute every time)
    np.save(OUT_DIR / "vorticity_gt.npy", vorticity)
    print(f"\n  cached vorticity to: {OUT_DIR / 'vorticity_gt.npy'}")

    print()
    print("=" * 60)
    print("Done. Open these files to verify the wake dynamics:")
    print(f"  {OUT_DIR / 'gt_u_velocity.mp4'}")
    print(f"  {OUT_DIR / 'gt_vorticity.mp4'}")
    print("=" * 60)
    print()
    print("You should see clear alternating red/blue patterns downstream of")
    print("the cylinder — that's the Karman vortex street at Re=100.")


if __name__ == "__main__":
    main()
