#!/usr/bin/env bash
# Run all experiments sequentially.
# Total wall time estimate (RTX 3060): ~8-10 hours.
# Intended for overnight runs.
#
# Usage:
#     bash scripts/run_all_experiments.sh
#
# To skip a run, comment it out below.

set -e  # abort on error

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "Running all experiments — start: $(date)"
echo "======================================================================"

# --- E1: Forward ---
python scripts/train.py --config configs/e1_forward.yaml

# --- E2: Inverse with N=5000 (baseline) ---
python scripts/train.py --config configs/e2_inverse_N5000.yaml

# --- E3: Data efficiency study (PINN) ---
python scripts/train.py --config configs/e3b_pinn_N2000.yaml
python scripts/train.py --config configs/e3c_pinn_N500.yaml
python scripts/train.py --config configs/e3d_pinn_N100.yaml

# --- E3: Data efficiency study (MLP baseline) ---
python scripts/train.py --config configs/e3e_mlp_N5000.yaml
python scripts/train.py --config configs/e3f_mlp_N500.yaml

echo ""
echo "======================================================================"
echo "All runs complete — end: $(date)"
echo "Now generating figures and animations..."
echo "======================================================================"

python scripts/make_figures.py
python scripts/make_animations.py

echo ""
echo "Done. See results/ and figures/ directories."
