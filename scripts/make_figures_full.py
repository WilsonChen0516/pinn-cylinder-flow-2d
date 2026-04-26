"""
Generate the E3 data-efficiency figure with both u and v components.

Two side-by-side subplots: left = u, right = v.
Compares PINN (with physics) vs MLP (data only).

Usage:
    python scripts/make_figures_full.py
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from src.visualization.styles import apply_style

apply_style()

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


# Same run mapping as make_figures.py
PINN_RUNS = {
    5000: "e2_inverse_N5000",   # E2 fix result (best PINN N=5000)
    2000: "e3b_pinn_N2000",
    500: "e3c_pinn_N500",
    100: "e3d_pinn_N100",
}
MLP_RUNS = {
    5000: "e3e_mlp_N5000",
    500: "e3f_mlp_N500",
}


def _load_metrics(run_name: str) -> dict | None:
    path = RESULTS_DIR / run_name / "metrics.json"
    if not path.exists():
        print(f"  SKIP {run_name}: metrics.json not found")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _gather_errors(runs: dict, field: str) -> tuple[list, list]:
    """field is 'u' or 'v'. Returns (n_list, err_list) sorted by n."""
    pairs = []
    metric_key = f"relative_l2_{field}"
    for n, run in runs.items():
        m = _load_metrics(run)
        if m is None:
            continue
        err = m["validation"].get(metric_key)
        if err is None:
            print(f"  WARN: {run} has no {metric_key}")
            continue
        pairs.append((n, err))
    pairs.sort()
    if not pairs:
        return [], []
    n_list, err_list = zip(*pairs)
    return list(n_list), list(err_list)


def plot_data_efficiency_uv(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    for ax, field, ylabel in [
        (axes[0], "u", "Relative L2 error  (u, streamwise)"),
        (axes[1], "v", "Relative L2 error  (v, cross-stream)"),
    ]:
        pinn_n, pinn_errs = _gather_errors(PINN_RUNS, field)
        mlp_n, mlp_errs = _gather_errors(MLP_RUNS, field)

        if pinn_n:
            ax.plot(pinn_n, pinn_errs, "o-",
                    label="PINN (with physics)",
                    linewidth=2, markersize=8)
        if mlp_n:
            ax.plot(mlp_n, mlp_errs, "s--",
                    label="MLP (data only)",
                    linewidth=2, markersize=8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of observation points  N")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{field}-component")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right")

    fig.suptitle("Data efficiency: PINN vs. pure data-driven MLP",
                 fontsize=13, y=1.02)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def print_summary_table() -> None:
    """Print a formatted summary table of all numbers."""
    print("\n" + "=" * 72)
    print("Data efficiency summary (Relative L2 error)")
    print("=" * 72)
    print(f"{'N':>6s}  {'Model':>6s}  {'L2(u)':>10s}  {'L2(v)':>10s}  {'L2(p)':>10s}")
    print("-" * 72)

    rows = []
    for n in sorted(set(list(PINN_RUNS.keys()) + list(MLP_RUNS.keys())),
                    reverse=True):
        if n in PINN_RUNS:
            m = _load_metrics(PINN_RUNS[n])
            if m:
                v = m["validation"]
                rows.append((n, "PINN",
                             v.get("relative_l2_u"), v.get("relative_l2_v"),
                             v.get("relative_l2_p")))
        if n in MLP_RUNS:
            m = _load_metrics(MLP_RUNS[n])
            if m:
                v = m["validation"]
                rows.append((n, "MLP",
                             v.get("relative_l2_u"), v.get("relative_l2_v"),
                             v.get("relative_l2_p")))

    for n, model, lu, lv, lp in rows:
        lu_s = f"{lu:.4e}" if lu is not None else "  —"
        lv_s = f"{lv:.4e}" if lv is not None else "  —"
        lp_s = f"{lp:.4e}" if lp is not None else "  —"
        print(f"{n:>6d}  {model:>6s}  {lu_s:>10s}  {lv_s:>10s}  {lp_s:>10s}")

    print("=" * 72)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating E3 data-efficiency figure (u and v subplots)...")
    plot_data_efficiency_uv(FIGURES_DIR / "fig_e3_data_efficiency_uv.png")

    print_summary_table()
    print(f"\nFigure saved to: {FIGURES_DIR / 'fig_e3_data_efficiency_uv.png'}")


if __name__ == "__main__":
    main()