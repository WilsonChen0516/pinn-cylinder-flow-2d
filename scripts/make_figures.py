"""
Generate all report figures from saved training results.

Reads results/<run>/metrics.json and history.json, produces:
  figures/fig_e1_loss.png
  figures/fig_e2_lambda_convergence.png
  figures/fig_e3_data_efficiency.png

Requires all relevant runs to have completed. Missing runs are skipped with a warning.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.visualization.plots import (
    plot_loss_curves, plot_lambda_convergence, plot_data_efficiency,
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


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


def make_e1_loss_figure():
    print("[E1] Generating loss curve figure...")
    hist = _load_history("e1_forward")
    if hist is None:
        return
    loss_hist = hist.get("loss", [])
    if not loss_hist:
        print("  no loss history")
        return

    series = {
        "step": [e["step"] for e in loss_hist],
        "total": [e["total"] for e in loss_hist],
        "ic": [e["ic"] for e in loss_hist],
        "bc": [e["bc"] for e in loss_hist],
        "pde": [e["pde"] for e in loss_hist],
    }
    plot_loss_curves(series, FIGURES_DIR / "fig_e1_loss.png", log_y=True)


def make_e2_lambda_figure():
    print("[E2] Generating lambda convergence figure...")
    hist = _load_history("e2_inverse_N5000")
    if hist is None:
        return
    lam = hist.get("lambdas", [])
    if not lam:
        print("  no lambda history")
        return

    steps = [e["step"] for e in lam]
    l1 = [e["lambda_1"] for e in lam]
    l2 = [e["lambda_2"] for e in lam]

    plot_lambda_convergence(
        steps=steps, lambda_1=l1, lambda_2=l2,
        true_lambda_1=1.0, true_lambda_2=0.01,
        out_path=FIGURES_DIR / "fig_e2_lambda_convergence.png",
        x_log=False,        # linear scale (was log)
        x_max=55000,        # show full Adam + L-BFGS span
    )


def make_e3_data_efficiency_figure():
    print("[E3] Generating data-efficiency figure...")

    pinn_runs = {
        5000: "e2_inverse_N5000",   # E2 full re-run (best PINN N=5000)
        2000: "e3b_pinn_N2000",
        500: "e3c_pinn_N500",
        100: "e3d_pinn_N100",
    }
    mlp_runs = {
        5000: "e3e_mlp_N5000",
        500: "e3f_mlp_N500",
    }

    pinn_n = []
    pinn_errs = []
    for n, run in sorted(pinn_runs.items()):
        m = _load_metrics(run)
        if m is None:
            continue
        pinn_n.append(n)
        pinn_errs.append(m["validation"]["relative_l2_u"])

    mlp_errs = {}
    for n, run in mlp_runs.items():
        m = _load_metrics(run)
        if m is None:
            continue
        mlp_errs[n] = m["validation"]["relative_l2_u"]

    if not pinn_n:
        print("  no PINN runs found, skipping")
        return
    if not mlp_errs:
        print("  WARNING: no MLP runs found, plotting PINN only")

    plot_data_efficiency(
        n_points_list=pinn_n,
        pinn_errors=pinn_errs,
        mlp_errors=mlp_errs,
        out_path=FIGURES_DIR / "fig_e3_data_efficiency.png",
    )


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating all report figures")
    print("=" * 60)

    make_e1_loss_figure()
    make_e2_lambda_figure()
    make_e3_data_efficiency_figure()

    print()
    print(f"Figures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()