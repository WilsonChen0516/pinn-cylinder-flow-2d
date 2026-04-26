"""
Unified matplotlib styling for reports and figures.

Import this once at the top of any plotting script:

    from src.visualization.styles import apply_style
    apply_style()
"""

import matplotlib
import matplotlib.pyplot as plt


REPORT_RCPARAMS = {
    # Figure
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "figure.autolayout": False,  # use constrained_layout explicitly
    "savefig.bbox": "tight",
    # Fonts — English-only stack for cross-platform consistency
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "DejaVu Sans",
        "Liberation Sans",
        "Bitstream Vera Sans",
        "sans-serif",
    ],
    "axes.unicode_minus": True,   # use real minus sign (Arial has it)
    # Sizes
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    # Lines
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    # Grid
    "axes.grid": False,
    "grid.alpha": 0.3,
    # Colors
    "axes.prop_cycle": plt.cycler(
        "color",
        ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"],
    ),
}


def apply_style() -> None:
    """Apply the project's matplotlib style."""
    matplotlib.rcParams.update(REPORT_RCPARAMS)


# Colormap conventions
CMAP_VELOCITY = "RdBu_r"   # signed quantity: u, v
CMAP_PRESSURE = "viridis"  # pressure
CMAP_VORTICITY = "RdBu_r"  # signed
CMAP_ERROR = "magma"       # absolute error (non-negative)