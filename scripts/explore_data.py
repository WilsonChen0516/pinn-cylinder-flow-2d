"""
Data exploration script.

Run after downloading data. Prints summary and saves a few diagnostic plots
to figures/exploration/ so you can verify the data looks right before
training anything.

Usage:
    python scripts/explore_data.py
"""

from pathlib import Path
import sys

# Add project root to path so we can import src.*
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.data.loader import CylinderWakeDataset

DATA_PATH = PROJECT_ROOT / "data" / "cylinder_nektar_wake.mat"
OUT_DIR = PROJECT_ROOT / "figures" / "exploration"


def plot_snapshot(ds: CylinderWakeDataset, time_index: int, out_path: Path) -> None:
    """Plot u, v, p at a single time step using triangulation."""
    snap = ds.snapshot(time_index)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fields = [("u", snap["u"], "RdBu_r"), ("v", snap["v"], "RdBu_r"), ("p", snap["p"], "viridis")]

    for ax, (name, field, cmap) in zip(axes, fields):
        sc = ax.tricontourf(snap["x"], snap["y"], field, levels=50, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(f"{name}  at t = {snap['t']:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(sc, ax=ax, shrink=0.7)

    fig.suptitle(f"Cylinder wake snapshot (time index {time_index})", fontsize=14)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_time_series_at_point(
    ds: CylinderWakeDataset, x_target: float, y_target: float, out_path: Path
) -> None:
    """Find nearest spatial point to (x_target, y_target) and plot u, v, p vs t."""
    dist = np.sqrt(
        (ds.X_star[:, 0] - x_target) ** 2 + (ds.X_star[:, 1] - y_target) ** 2
    )
    idx = int(np.argmin(dist))
    actual_x, actual_y = ds.X_star[idx]
    print(f"  nearest point to ({x_target}, {y_target}): ({actual_x:.3f}, {actual_y:.3f})")

    u_t = ds.U_star[idx, 0, :]
    v_t = ds.U_star[idx, 1, :]
    p_t = ds.p_star[idx, :]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(ds.t_star, u_t)
    axes[0].set_ylabel("u")
    axes[0].grid(alpha=0.3)
    axes[1].plot(ds.t_star, v_t)
    axes[1].set_ylabel("v")
    axes[1].grid(alpha=0.3)
    axes[2].plot(ds.t_star, p_t)
    axes[2].set_ylabel("p")
    axes[2].set_xlabel("t")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Time series at (x, y) ≈ ({actual_x:.2f}, {actual_y:.2f})", fontsize=13)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_spatial_grid(ds: CylinderWakeDataset, out_path: Path) -> None:
    """Scatter plot of all spatial sampling points."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(ds.X_star[:, 0], ds.X_star[:, 1], s=1, c="k", alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Spatial grid ({ds.N} points)")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main() -> None:
    print("=" * 60)
    print("PINN Cylinder Wake — Data Exploration")
    print("=" * 60)

    ds = CylinderWakeDataset(DATA_PATH)
    print()
    print(ds.summary())
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # A few time steps to eyeball vortex shedding
    print("Generating snapshot plots...")
    for t_idx in [0, 50, 100, 150, 199]:
        plot_snapshot(ds, t_idx, OUT_DIR / f"snapshot_t{t_idx:03d}.png")

    # Time series at a point in the wake
    print("\nGenerating time-series plots...")
    plot_time_series_at_point(ds, x_target=3.0, y_target=0.0, out_path=OUT_DIR / "timeseries_x3_y0.png")
    plot_time_series_at_point(ds, x_target=5.0, y_target=0.5, out_path=OUT_DIR / "timeseries_x5_y05.png")

    # Spatial grid
    print("\nGenerating spatial grid plot...")
    plot_spatial_grid(ds, OUT_DIR / "spatial_grid.png")

    print()
    print("=" * 60)
    print(f"All exploration figures saved to: {OUT_DIR}")
    print("=" * 60)
    print()
    print("What to look for:")
    print("  - snapshot_t000.png : 初始時刻流場，應有輕微擾動")
    print("  - snapshot_t150.png : 後期流場，應有清楚的 Karman vortex street（紅藍交替）")
    print("  - timeseries_x3_y0  : u 應有振盪（~5-6 秒週期）")
    print("  - spatial_grid.png  : 5000 個點散布在矩形域內")


if __name__ == "__main__":
    main()
