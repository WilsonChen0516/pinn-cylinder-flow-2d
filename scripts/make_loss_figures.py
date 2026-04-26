"""
Generate supplementary loss-related figures.

  - fig_e2_loss.png        E2 inverse: total / data / pde loss curves
  - fig_e3_loss_summary.png  Bar chart of final loss per E3 run

These complement fig_e1_loss.png (which already exists from make_figures.py).

Usage:
    python scripts/make_loss_figures.py
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.styles import apply_style

apply_style()

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


# E1 forward run (A baseline)
E1_RUN = "e1_ablation_A_baseline"

# E2 final run name (full re-run)
E2_RUN = "e2_inverse_N5000"

# All E3 runs to summarize
E3_RUNS = [
    ("e2_inverse_N5000",     "PINN N=5000"),
    ("e3b_pinn_N2000",       "PINN N=2000"),
    ("e3c_pinn_N500",        "PINN N=500"),
    ("e3d_pinn_N100",        "PINN N=100"),
    ("e3e_mlp_N5000",        "MLP N=5000"),
    ("e3f_mlp_N500",         "MLP N=500"),
]


def _load_history(run_name: str) -> dict | None:
    path = RESULTS_DIR / run_name / "history.json"
    if not path.exists():
        print(f"  SKIP {run_name}: history.json not found")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_metrics(run_name: str) -> dict | None:
    path = RESULTS_DIR / run_name / "metrics.json"
    if not path.exists():
        print(f"  SKIP {run_name}: metrics.json not found")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# E1 Forward Loss curves (IC, BC, PDE)
# -----------------------------------------------------------------------------

def make_e1_loss_figure():
    print(f"[E1] Generating loss curve figure from {E1_RUN}...")
    hist = _load_history(E1_RUN)
    if hist is None:
        return
    loss_hist = hist.get("loss", [])
    if not loss_hist:
        print("  no loss history")
        return

    steps = np.array([e["step"] for e in loss_hist])
    total = np.array([e["total"] for e in loss_hist])
    pde = np.array([e["pde"] for e in loss_hist])
    ic = np.array([e.get("ic", 0) for e in loss_hist])
    bc = np.array([e.get("bc", 0) for e in loss_hist])

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(steps, total, label="Total", linewidth=2, color="#1f77b4")
    ax.plot(steps, ic, label="IC (initial condition)", linewidth=1.5, alpha=0.8, color="#d62728")
    ax.plot(steps, bc, label="BC (boundary condition)", linewidth=1.5, alpha=0.8, color="#2ca02c")
    ax.plot(steps, pde, label="PDE residual", linewidth=1.5, alpha=0.8, color="#ff7f0e")

    # Mark Adam → L-BFGS transition
    adam_end = 50000
    if any(s > adam_end for s in steps):
        ax.axvline(adam_end, color="gray", linestyle=":",
                   alpha=0.7, label=f"Adam → L-BFGS")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("E1 Forward — Training Loss")

    out_path = FIGURES_DIR / "fig_e1_loss.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# -----------------------------------------------------------------------------
# E2 Loss curves
# -----------------------------------------------------------------------------

def make_e2_loss_figure():
    print(f"[E2] Generating loss curve figure from {E2_RUN}...")
    hist = _load_history(E2_RUN)
    if hist is None:
        return
    loss_hist = hist.get("loss", [])
    if not loss_hist:
        print("  no loss history")
        return

    steps = np.array([e["step"] for e in loss_hist])
    total = np.array([e["total"] for e in loss_hist])
    data = np.array([e["data"] for e in loss_hist])
    pde = np.array([e["pde"] for e in loss_hist])

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.plot(steps, total, label="Total", linewidth=2)
    ax.plot(steps, data, label="Data (u, v MSE)", linewidth=1.5, alpha=0.8)
    ax.plot(steps, pde, label="PDE residual", linewidth=1.5, alpha=0.8)

    # Mark Adam → L-BFGS transition if we can detect it
    # Heuristic: find the step where the spacing changes (Adam logs every 100,
    # L-BFGS logs every 100 too but step values come from different phases)
    # Just look at metrics.json for total step counts
    metrics = _load_metrics(E2_RUN)
    if metrics:
        # Adam steps are usually 50000; if final_step > 50000, L-BFGS started at 50000
        adam_end = 50000
        if any(s > adam_end for s in steps):
            ax.axvline(adam_end, color="gray", linestyle=":",
                       alpha=0.7, label=f"Adam → L-BFGS")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(f"E2 Inverse — Training Loss ({E2_RUN})")

    out_path = FIGURES_DIR / "fig_e2_loss.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# -----------------------------------------------------------------------------
# E3 Loss summary bar chart
# -----------------------------------------------------------------------------

def make_e3_loss_summary():
    print(f"[E3] Generating loss summary bar chart...")

    rows = []
    for run, label in E3_RUNS:
        hist = _load_history(run)
        if hist is None:
            continue
        loss_hist = hist.get("loss", [])
        if not loss_hist:
            continue
        last = loss_hist[-1]
        rows.append({
            "label": label,
            "run": run,
            "total": last["total"],
            "data": last["data"],
            "pde": last.get("pde", 0),
        })

    if not rows:
        print("  no runs found")
        return

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.35

    data_vals = [r["data"] for r in rows]
    pde_vals = [r["pde"] for r in rows]

    ax.bar(x - width/2, data_vals, width, label="Data loss", color="#1f77b4")
    ax.bar(x + width/2, pde_vals, width, label="PDE loss", color="#d62728")

    ax.set_yscale("log")
    ax.set_ylabel("Final loss (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.set_title("E3 Final loss components per run")

    # Annotate bars with values
    for i, r in enumerate(rows):
        if r["data"] > 0:
            ax.text(i - width/2, r["data"], f"{r['data']:.1e}",
                    ha="center", va="bottom", fontsize=8, rotation=0)
        if r["pde"] > 0:
            ax.text(i + width/2, r["pde"], f"{r['pde']:.1e}",
                    ha="center", va="bottom", fontsize=8, rotation=0)

    out_path = FIGURES_DIR / "fig_e3_loss_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# -----------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating supplementary loss figures")
    print("=" * 60)

    make_e1_loss_figure()
    make_e2_loss_figure()
    make_e3_loss_summary()

    print()
    print(f"Output: {FIGURES_DIR}")


if __name__ == "__main__":
    main()