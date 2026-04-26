"""
Quick test: Does lowering L-BFGS lr help?

Loads E2's final checkpoint, runs L-BFGS with different lr values,
reports whether loss changes at all.

Usage:
    python scripts/test_lbfgs_lr.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.loader import CylinderWakeDataset
from src.models.pinn import PINN
from src.physics.navier_stokes import ns_residual_2d, velocity_from_streamfunction
from src.utils.seed import set_seed


DATA_PATH = PROJECT_ROOT / "data" / "cylinder_nektar_wake.mat"
CKPT_DIR = PROJECT_ROOT / "results" / "e2_inverse_N5000" / "checkpoints"


def load_model(device: str) -> PINN:
    ds = CylinderWakeDataset(DATA_PATH)
    lb, ub = ds.domain.as_lower_upper()

    model = PINN(lb=lb, ub=ub, hidden_layers=8, neurons_per_layer=20,
                 activation="tanh", learn_lambdas=True).to(device)

    final = sorted(CKPT_DIR.glob("*_final.pt"))
    if not final:
        raise FileNotFoundError("No final checkpoint found")
    payload = torch.load(final[-1], map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    return model, ds


def make_batches(ds: CylinderWakeDataset, device: str):
    import numpy as np
    obs = ds.random_subsample(5000, seed=42)
    def to_t(arr):
        return torch.from_numpy(arr.astype(np.float32)).to(device)
    return {
        "x": to_t(obs["x"]), "y": to_t(obs["y"]), "t": to_t(obs["t"]),
        "u": to_t(obs["u"]), "v": to_t(obs["v"]),
    }


def compute_loss(model, batch):
    x = batch["x"].detach().requires_grad_(True)
    y = batch["y"].detach().requires_grad_(True)
    t = batch["t"].detach().requires_grad_(True)

    psi, p = model(x, y, t)
    u, v, f_u, f_v = ns_residual_2d(
        psi, p, x, y, t,
        model.effective_lambda_1, model.effective_lambda_2,
    )

    L_data = torch.mean((u - batch["u"])**2) + torch.mean((v - batch["v"])**2)
    L_pde = torch.mean(f_u**2) + torch.mean(f_v**2)
    return L_data + L_pde, L_data, L_pde


def test_lr(lr: float, steps: int, model_template, ds, device: str):
    # Deep copy model so each test starts from same point
    import copy
    model = copy.deepcopy(model_template)
    batch = make_batches(ds, device)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr,
        max_iter=20, max_eval=25,
        history_size=50,
        tolerance_grad=1e-8, tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    loss_before, _, _ = compute_loss(model, batch)
    loss_before = loss_before.item()

    for i in range(steps):
        def closure():
            optimizer.zero_grad()
            loss, _, _ = compute_loss(model, batch)
            loss.backward()
            return loss
        optimizer.step(closure)

    loss_after, data_after, pde_after = compute_loss(model, batch)
    l1 = model.effective_lambda_1.item()
    l2 = model.effective_lambda_2.item()

    return {
        "lr": lr,
        "loss_before": loss_before,
        "loss_after": loss_after.item(),
        "change_pct": 100 * (loss_after.item() - loss_before) / (abs(loss_before) + 1e-12),
        "lambda_1": l1,
        "lambda_2": l2,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)

    print("Loading E2 checkpoint...")
    model, ds = load_model(device)
    print(f"  λ₁ = {model.effective_lambda_1.item():.6f}")
    print(f"  λ₂ = {model.effective_lambda_2.item():.6f}")
    print()

    lrs = [1.0, 0.1, 0.01, 0.001]
    steps = 200  # 200 steps per test, total ~5 min

    print(f"Testing {len(lrs)} learning rates × {steps} steps each")
    print(f"{'lr':>8s}  {'loss_before':>12s}  {'loss_after':>12s}  {'change%':>10s}  {'λ₁':>8s}  {'λ₂':>10s}")
    print("-" * 72)

    for lr in lrs:
        result = test_lr(lr, steps, model, ds, device)
        print(f"{result['lr']:8.4f}  "
              f"{result['loss_before']:12.6e}  "
              f"{result['loss_after']:12.6e}  "
              f"{result['change_pct']:+10.4f}%  "
              f"{result['lambda_1']:8.6f}  "
              f"{result['lambda_2']:10.8f}")

    print()
    print("Interpretation:")
    print("  change% ≈ 0  → L-BFGS cannot improve from this point at any lr")
    print("  change% < 0  → L-BFGS found improvement (lower = better)")
    print("  change% > 0  → L-BFGS made it worse (unstable)")


if __name__ == "__main__":
    main()