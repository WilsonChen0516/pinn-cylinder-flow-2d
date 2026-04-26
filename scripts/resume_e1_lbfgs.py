"""
Resume an E1 ablation run's L-BFGS phase from the post-Adam checkpoint.

Allows trying different L-BFGS learning rates without re-running 50k Adam steps.
Designed for diagnosing why L-BFGS stalls on certain weight configurations.

Usage:
    # Default: A baseline run, lr=1.0, 1000 steps
    python scripts/resume_e1_lbfgs.py

    # Specify run, lr, steps
    python scripts/resume_e1_lbfgs.py --run e1_ablation_A_baseline --lr 1.0 --steps 1000
    python scripts/resume_e1_lbfgs.py --run e1_ablation_A_baseline --lr 0.5 --steps 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.loader import CylinderWakeDataset
from src.data.sampler import latin_hypercube
from src.models.pinn import PINN
from src.physics.navier_stokes import ns_residual_2d
from src.evaluation.validator import Validator
from src.training.callbacks import save_final_metrics
from src.utils.logging import get_logger
from src.utils.seed import set_seed


DATA_PATH = PROJECT_ROOT / "data" / "cylinder_nektar_wake.mat"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_pinn_from_run(run_name: str, ds: CylinderWakeDataset, device: str):
    ckpt_dir = RESULTS_DIR / run_name / "checkpoints"

    # Prefer step_0050000.pt (post-Adam, before L-BFGS) if available
    adam_end = ckpt_dir / "step_0050000.pt"
    if adam_end.exists():
        ckpt_path = adam_end
        print(f"  using post-Adam checkpoint: {ckpt_path}")
    else:
        # Fall back to final
        final = sorted(ckpt_dir.glob("*_final.pt"))
        if not final:
            raise FileNotFoundError(f"No checkpoint in {ckpt_dir}")
        ckpt_path = final[-1]
        print(f"  using final checkpoint: {ckpt_path}")
        print(f"  (note: L-BFGS already ran; results should be identical to Adam end)")

    lb, ub = ds.domain.as_lower_upper()
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = PINN(
        lb=lb, ub=ub,
        hidden_layers=8, neurons_per_layer=20, activation="tanh",
        learn_lambdas=False,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model


def prepare_batches_forward(ds: CylinderWakeDataset, cfg: dict, device: str):
    """Replicate the forward problem batches from train.py."""
    n_ic = cfg["data"]["n_ic_points"]
    n_bc_spatial = cfg["data"]["n_bc_spatial"]
    n_bc_times = cfg["data"]["n_bc_times"]
    n_collocation = cfg["data"]["n_collocation"]

    rng = np.random.default_rng(42)

    # IC batch (t=0)
    snap0 = ds.snapshot(0)
    idx_ic = rng.choice(ds.N, size=min(n_ic, ds.N), replace=False)
    ic_batch = {
        "x": torch.from_numpy(snap0["x"][idx_ic].reshape(-1, 1).astype(np.float32)).to(device),
        "y": torch.from_numpy(snap0["y"][idx_ic].reshape(-1, 1).astype(np.float32)).to(device),
        "t": torch.full((len(idx_ic), 1), 0.0, dtype=torch.float32, device=device),
        "u": torch.from_numpy(snap0["u"][idx_ic].reshape(-1, 1).astype(np.float32)).to(device),
        "v": torch.from_numpy(snap0["v"][idx_ic].reshape(-1, 1).astype(np.float32)).to(device),
    }

    # BC batches
    d = ds.domain
    x_pts = ds.X_star[:, 0]
    y_pts = ds.X_star[:, 1]
    tol_x = (d.x_max - d.x_min) * 0.01
    tol_y = (d.y_max - d.y_min) * 0.01

    def _pick_edge(mask, n):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = rng.choice(idx, size=n, replace=False)
        return idx

    edges = {
        "left":   _pick_edge(x_pts < d.x_min + tol_x, n_bc_spatial),
        "right":  _pick_edge(x_pts > d.x_max - tol_x, n_bc_spatial),
        "bottom": _pick_edge(y_pts < d.y_min + tol_y, n_bc_spatial),
        "top":    _pick_edge(y_pts > d.y_max - tol_y, n_bc_spatial),
    }

    bc_xs, bc_ys, bc_ts, bc_us, bc_vs = [], [], [], [], []
    times_for_bc = rng.choice(ds.T, size=n_bc_times, replace=False)
    for edge_name, edge_idx in edges.items():
        for tk in times_for_bc:
            for sp in edge_idx:
                bc_xs.append(ds.X_star[sp, 0])
                bc_ys.append(ds.X_star[sp, 1])
                bc_ts.append(ds.t_star[tk])
                bc_us.append(ds.U_star[sp, 0, tk])
                bc_vs.append(ds.U_star[sp, 1, tk])

    bc_batch = {
        "x": torch.tensor(bc_xs, dtype=torch.float32, device=device).reshape(-1, 1),
        "y": torch.tensor(bc_ys, dtype=torch.float32, device=device).reshape(-1, 1),
        "t": torch.tensor(bc_ts, dtype=torch.float32, device=device).reshape(-1, 1),
        "u": torch.tensor(bc_us, dtype=torch.float32, device=device).reshape(-1, 1),
        "v": torch.tensor(bc_vs, dtype=torch.float32, device=device).reshape(-1, 1),
    }

    # Collocation
    col = latin_hypercube(
        n_collocation,
        lb=np.array([d.x_min, d.y_min, d.t_min]),
        ub=np.array([d.x_max, d.y_max, d.t_max]),
        seed=42,
    )
    col_batch = {
        "x": torch.from_numpy(col[:, 0:1].astype(np.float32)).to(device),
        "y": torch.from_numpy(col[:, 1:2].astype(np.float32)).to(device),
        "t": torch.from_numpy(col[:, 2:3].astype(np.float32)).to(device),
    }

    return ic_batch, bc_batch, col_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="e1_ablation_A_baseline",
                        help="Run name to resume from")
    parser.add_argument("--lr", type=float, default=1.0,
                        help="L-BFGS learning rate to try")
    parser.add_argument("--steps", type=int, default=1000,
                        help="L-BFGS steps")
    parser.add_argument("--ic-weight", type=float, default=1.0)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--pde-weight", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--history-size", type=int, default=50)
    args = parser.parse_args()

    out_dir = RESULTS_DIR / f"{args.run}_lbfgs_lr{args.lr:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("lbfgs_resume", log_file=out_dir / "logs" / "training.log")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(42)
    logger.info(f"Device: {device}")
    logger.info(f"Resuming from: {args.run}")
    logger.info(f"L-BFGS lr = {args.lr}, steps = {args.steps}")
    logger.info(f"Loss weights: pde={args.pde_weight}, ic={args.ic_weight}, bc={args.bc_weight}")

    # Load data and dataset config
    ds = CylinderWakeDataset(DATA_PATH)

    # Load original config to get data params
    src_config = RESULTS_DIR / args.run / "config_snapshot.yaml"
    if src_config.exists():
        from src.utils.config import load_config
        # We don't extend because config_snapshot is fully resolved
        import yaml
        with open(src_config) as f:
            cfg = yaml.safe_load(f)
    else:
        # Use defaults
        cfg = {"data": {"n_ic_points": 5000, "n_bc_spatial": 100,
                        "n_bc_times": 50, "n_collocation": 20000}}

    # Load model
    model = load_pinn_from_run(args.run, ds, device)

    # Prepare batches
    ic_batch, bc_batch, col_batch = prepare_batches_forward(ds, cfg, device)

    # Validate before L-BFGS
    validator = Validator(ds, device=device)
    val_before = validator.evaluate(model, model_type="pinn")
    logger.info(f"Before L-BFGS: L2(u)={val_before.relative_l2_u:.4e}  "
                f"L2(v)={val_before.relative_l2_v:.4e}  "
                f"L2(p)={val_before.relative_l2_p:.4e}")

    # --- Compute loss ---
    def compute_loss():
        # IC loss
        x = ic_batch["x"].detach().requires_grad_(True)
        y = ic_batch["y"].detach().requires_grad_(True)
        t = ic_batch["t"].detach().requires_grad_(True)
        psi, _ = model(x, y, t)
        from src.physics.navier_stokes import velocity_from_streamfunction
        u, v = velocity_from_streamfunction(psi, x, y)
        L_ic = torch.mean((u - ic_batch["u"])**2) + torch.mean((v - ic_batch["v"])**2)

        # BC loss
        x = bc_batch["x"].detach().requires_grad_(True)
        y = bc_batch["y"].detach().requires_grad_(True)
        t = bc_batch["t"].detach().requires_grad_(True)
        psi, _ = model(x, y, t)
        u, v = velocity_from_streamfunction(psi, x, y)
        L_bc = torch.mean((u - bc_batch["u"])**2) + torch.mean((v - bc_batch["v"])**2)

        # PDE loss
        x = col_batch["x"].detach().requires_grad_(True)
        y = col_batch["y"].detach().requires_grad_(True)
        t = col_batch["t"].detach().requires_grad_(True)
        psi, p = model(x, y, t)
        u, v, f_u, f_v = ns_residual_2d(psi, p, x, y, t,
                                         lambda_1=1.0, lambda_2=0.01)
        L_pde = torch.mean(f_u**2) + torch.mean(f_v**2)

        total = (args.pde_weight * L_pde +
                 args.ic_weight * L_ic +
                 args.bc_weight * L_bc)
        return total, L_ic, L_bc, L_pde

    # --- L-BFGS ---
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=args.lr,
        max_iter=args.max_iter,
        max_eval=int(1.25 * args.max_iter),
        history_size=args.history_size,
        tolerance_grad=1e-8,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    logger.info(f"=== L-BFGS phase: {args.steps} steps, lr={args.lr} ===")
    t0 = time.time()

    for i in range(args.steps):
        def closure():
            optimizer.zero_grad()
            loss, _, _, _ = compute_loss()
            if torch.isfinite(loss):
                loss.backward()
            return loss

        optimizer.step(closure)

        if (i + 1) % 100 == 0:
            total, L_ic, L_bc, L_pde = compute_loss()
            elapsed = time.time() - t0
            logger.info(
                f"[lbfgs {i+1:5d}/{args.steps}]  "
                f"total={total.item():.3e}  "
                f"pde={L_pde.item():.3e}  "
                f"ic={L_ic.item():.3e}  "
                f"bc={L_bc.item():.3e}  "
                f"({elapsed:.0f}s)"
            )

    elapsed = time.time() - t0
    logger.info(f"L-BFGS done in {elapsed:.1f}s")

    # Validate after
    val_after = validator.evaluate(model, model_type="pinn")
    logger.info(f"After  L-BFGS: L2(u)={val_after.relative_l2_u:.4e}  "
                f"L2(v)={val_after.relative_l2_v:.4e}  "
                f"L2(p)={val_after.relative_l2_p:.4e}")

    # Save
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": args.steps,
        "model_state_dict": model.state_dict(),
    }, out_dir / "checkpoints" / "step_final.pt")

    metrics = {
        "run_name": out_dir.name,
        "note": f"Resumed from {args.run}, L-BFGS lr={args.lr} x {args.steps} steps",
        "lbfgs_lr": args.lr,
        "lbfgs_steps": args.steps,
        "loss_weights": {"pde": args.pde_weight,
                         "ic": args.ic_weight,
                         "bc": args.bc_weight},
        "validation_before": val_before.as_dict(),
        "validation_after": val_after.as_dict(),
    }
    save_final_metrics(metrics, out_dir)

    # Print delta
    print()
    print(f"=== Result summary ===")
    print(f"  before: L2(u)={val_before.relative_l2_u*100:.3f}%  "
          f"L2(v)={val_before.relative_l2_v*100:.3f}%  "
          f"L2(p)={val_before.relative_l2_p*100:.3f}%")
    print(f"  after:  L2(u)={val_after.relative_l2_u*100:.3f}%  "
          f"L2(v)={val_after.relative_l2_v*100:.3f}%  "
          f"L2(p)={val_after.relative_l2_p*100:.3f}%")
    print(f"  output: {out_dir}")


if __name__ == "__main__":
    main()