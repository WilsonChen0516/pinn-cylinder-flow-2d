"""
Main training entry point.

Usage:
    python scripts/train.py --config configs/e1_forward.yaml
    python scripts/train.py --config configs/e2_inverse_N5000.yaml

The config file determines:
  - Experiment type: "forward" | "inverse" | "mlp_baseline"
  - Data sampling (N observations, collocation count, IC/BC)
  - Model architecture
  - Training schedule (Adam steps, L-BFGS steps)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.data.loader import CylinderWakeDataset
from src.data.sampler import latin_hypercube, sample_initial_condition, sample_boundary_2d_rect
from src.models.pinn import PINN
from src.models.mlp import MLP
from src.physics.navier_stokes import velocity_from_streamfunction
from src.training.losses import PINNLoss, MLPDataLoss, LossWeights
from src.training.trainer import Trainer
from src.training.callbacks import (
    CheckpointCallback, ValidationCallback, LambdaTrackingCallback,
    LossHistoryCallback, SnapshotCallback,
    save_history, save_final_metrics,
)
from src.evaluation.validator import Validator
from src.evaluation.metrics import parameter_error_pct
from src.utils.config import load_config, save_config_snapshot
from src.utils.logging import get_logger
from src.utils.seed import set_seed


# -----------------------------------------------------------------------------

TRUE_LAMBDA_1 = 1.0
TRUE_LAMBDA_2 = 0.01   # = 1 / Re, with Re = 100


# -----------------------------------------------------------------------------
# Batch preparation
# -----------------------------------------------------------------------------

def _to_tensor_dict(d: dict, device: str) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.dtype != object:
            out[k] = torch.from_numpy(v.astype(np.float32)).to(device)
        else:
            out[k] = v
    return out


def prepare_batches_forward(
    ds: CylinderWakeDataset, cfg: dict, device: str, seed: int
) -> dict:
    """
    Forward problem: IC + BC from Raissi data, PDE collocation from LHS.
    """
    lb, ub = ds.domain.as_lower_upper()

    # --- Initial condition (t = t_min, sample from Raissi t=0 snapshot) ---
    n_ic = cfg["data"].get("n_ic_points", 5000)
    snap0 = ds.snapshot(0)
    # Use all spatial points at t=0 if n_ic >= N, else random subset
    if n_ic >= ds.N:
        idx_ic = np.arange(ds.N)
    else:
        rng = np.random.default_rng(seed)
        idx_ic = rng.choice(ds.N, size=n_ic, replace=False)
    ic_batch = {
        "x": snap0["x"][idx_ic][:, None],
        "y": snap0["y"][idx_ic][:, None],
        "t": np.full((len(idx_ic), 1), snap0["t"], dtype=np.float32),
        "u": snap0["u"][idx_ic][:, None],
        "v": snap0["v"][idx_ic][:, None],
    }

    # --- Boundary condition: take all edge spatial points at each time ---
    n_bc_spatial = cfg["data"].get("n_bc_spatial", 100)   # per edge
    n_bc_times = cfg["data"].get("n_bc_times", 50)        # time samples per edge
    bc_batch = _build_bc_from_dataset(ds, n_bc_spatial, n_bc_times, seed)

    # --- Collocation: LHS in (x, y, t) ---
    n_col = cfg["data"].get("n_collocation", 20000)
    col_pts = latin_hypercube(n_col, lb, ub, seed=seed + 1)
    col_batch = {
        "x": col_pts[:, 0:1],
        "y": col_pts[:, 1:2],
        "t": col_pts[:, 2:3],
    }

    return {
        "ic": _to_tensor_dict(ic_batch, device),
        "bc": _to_tensor_dict(bc_batch, device),
        "col": _to_tensor_dict(col_batch, device),
    }


def _build_bc_from_dataset(
    ds: CylinderWakeDataset, n_per_edge: int, n_times: int, seed: int
) -> dict:
    """Extract boundary points (x or y at extremes) from the Raissi dataset."""
    rng = np.random.default_rng(seed + 7)

    d = ds.domain
    tol_x = 0.02 * (d.x_max - d.x_min)
    tol_y = 0.02 * (d.y_max - d.y_min)

    x_pts = ds.X_star[:, 0]
    y_pts = ds.X_star[:, 1]

    def _pick_edge(mask, n):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.array([], dtype=int)
        if len(idx) < n:
            return idx
        return rng.choice(idx, size=n, replace=False)

    left = _pick_edge(x_pts < d.x_min + tol_x, n_per_edge)
    right = _pick_edge(x_pts > d.x_max - tol_x, n_per_edge)
    bottom = _pick_edge(y_pts < d.y_min + tol_y, n_per_edge)
    top = _pick_edge(y_pts > d.y_max - tol_y, n_per_edge)
    spatial_idx = np.concatenate([left, right, bottom, top])

    # For each spatial boundary point, pick n_times time steps
    t_idx = rng.choice(ds.T, size=n_times, replace=False)

    xs, ys, ts, us, vs = [], [], [], [], []
    for si in spatial_idx:
        for ti in t_idx:
            xs.append(ds.X_star[si, 0])
            ys.append(ds.X_star[si, 1])
            ts.append(ds.t_star[ti])
            us.append(ds.U_star[si, 0, ti])
            vs.append(ds.U_star[si, 1, ti])

    return {
        "x": np.array(xs, dtype=np.float32)[:, None],
        "y": np.array(ys, dtype=np.float32)[:, None],
        "t": np.array(ts, dtype=np.float32)[:, None],
        "u": np.array(us, dtype=np.float32)[:, None],
        "v": np.array(vs, dtype=np.float32)[:, None],
    }


def prepare_batches_inverse(
    ds: CylinderWakeDataset, cfg: dict, device: str, seed: int
) -> dict:
    """
    Inverse problem: sparse (u, v) observations serve as both data source
    and collocation points (following Raissi).
    """
    n_obs = cfg["data"]["n_observations"]
    obs = ds.random_subsample(n_obs, seed=seed)

    obs_batch = {
        "x": obs["x"], "y": obs["y"], "t": obs["t"],
        "u": obs["u"], "v": obs["v"],
    }
    # PDE enforced on same points
    col_batch = {"x": obs["x"], "y": obs["y"], "t": obs["t"]}

    return {
        "obs": _to_tensor_dict(obs_batch, device),
        "col": _to_tensor_dict(col_batch, device),
    }


def prepare_batches_mlp(
    ds: CylinderWakeDataset, cfg: dict, device: str, seed: int
) -> dict:
    n_obs = cfg["data"]["n_observations"]
    obs = ds.random_subsample(n_obs, seed=seed)
    return {
        "obs": _to_tensor_dict({
            "x": obs["x"], "y": obs["y"], "t": obs["t"],
            "u": obs["u"], "v": obs["v"],
        }, device),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    # --- Load config ---
    cfg = load_config(args.config)
    run_name = cfg["experiment"]["name"]
    exp_type = cfg["experiment"]["type"]   # "forward" | "inverse" | "mlp_baseline"

    out_dir = PROJECT_ROOT / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, out_dir / "config_snapshot.yaml")

    # --- Logger ---
    logger = get_logger(run_name, log_file=out_dir / "logs" / "training.log")
    logger.info(f"Starting run: {run_name}  (type: {exp_type})")

    # --- Device & seed ---
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load data ---
    data_path = PROJECT_ROOT / cfg["data"]["path"]
    ds = CylinderWakeDataset(data_path)
    logger.info(ds.summary())
    lb, ub = ds.domain.as_lower_upper()

    # --- Build model ---
    mcfg = cfg["model"]
    if exp_type in ("forward", "inverse"):
        model = PINN(
            lb=lb, ub=ub,
            hidden_layers=mcfg["hidden_layers"],
            neurons_per_layer=mcfg["neurons_per_layer"],
            activation=mcfg.get("activation", "tanh"),
            learn_lambdas=(exp_type == "inverse"),
            init_lambda_1=cfg.get("inverse", {}).get("init_lambda_1", 0.0),
            init_lambda_2=cfg.get("inverse", {}).get("init_lambda_2", 0.0),
        )
    elif exp_type == "mlp_baseline":
        model = MLP(
            lb=lb, ub=ub,
            hidden_layers=mcfg["hidden_layers"],
            neurons_per_layer=mcfg["neurons_per_layer"],
            activation=mcfg.get("activation", "tanh"),
        )
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

    logger.info(f"Model: {model.__class__.__name__}  params={sum(p.numel() for p in model.parameters())}")

    # --- Prepare batches ---
    seed = cfg.get("seed", 42)
    if exp_type == "forward":
        batches = prepare_batches_forward(ds, cfg, device, seed)
    elif exp_type == "inverse":
        batches = prepare_batches_inverse(ds, cfg, device, seed)
    else:  # mlp_baseline
        batches = prepare_batches_mlp(ds, cfg, device, seed)

    for k, v in batches.items():
        n = len(v["x"]) if "x" in v else "?"
        logger.info(f"Batch [{k}]: {n} points")

    # --- Build loss ---
    if exp_type == "mlp_baseline":
        loss_fn = MLPDataLoss()
    else:
        w = cfg.get("loss_weights", {})
        weights = LossWeights(
            data=w.get("data", 1.0),
            pde=w.get("pde", 1.0),
            ic=w.get("ic", 1.0),
            bc=w.get("bc", 1.0),
        )
        loss_fn = PINNLoss(weights=weights)

    # --- Validator & callbacks ---
    model_type_for_val = "mlp" if exp_type == "mlp_baseline" else "pinn"
    validator = Validator(ds, device=device, max_points=20000)

    callbacks = [
        LossHistoryCallback(every=cfg.get("log_every_history", 100)),
        ValidationCallback(validator, every=cfg.get("val_every", 5000),
                           model_type=model_type_for_val),
        CheckpointCallback(out_dir / "checkpoints",
                           every=cfg.get("ckpt_every", 5000), keep_last=3),
    ]
    if exp_type == "inverse":
        callbacks.append(LambdaTrackingCallback(every=cfg.get("lambda_track_every", 100)))

    # Snapshot for A2 animation (optional; only if requested)
    if cfg.get("snapshot_for_animation", False):
        t_snap = cfg.get("snapshot_t", float(ds.t_star[len(ds.t_star) // 2]))
        callbacks.append(SnapshotCallback(
            grid_x=ds.X_star[:, 0],
            grid_y=ds.X_star[:, 1],
            grid_t=t_snap,
            out_dir=out_dir / "snapshots",
            every=cfg.get("snapshot_every", 2000),
            model_type=model_type_for_val,
        ))

    # --- Train ---
    trainer = Trainer(model, loss_fn, batches, device=device,
                      callbacks=callbacks, logger=logger)

    tcfg = cfg["training"]
    t_start = time.time()

    trainer.train_adam(
        n_steps=tcfg["adam_steps"],
        lr=tcfg.get("adam_lr", 1e-3),
        lr_final=tcfg.get("adam_lr_final", 1e-4),
        log_every=tcfg.get("log_every", 500),
    )

    if tcfg.get("lbfgs_steps", 0) > 0:
        trainer.train_lbfgs(
            max_steps=tcfg["lbfgs_steps"],
            lr=tcfg.get("lbfgs_lr", 1.0),
            history_size=tcfg.get("lbfgs_history_size", 50),
            max_iter_per_step=tcfg.get("lbfgs_max_iter", 20),
            log_every=tcfg.get("lbfgs_log_every", 100),
        )

    trainer.on_train_end()
    total_time = time.time() - t_start
    logger.info(f"Training done in {total_time:.1f}s ({total_time / 60:.1f} min)")

    # --- Final validation ---
    final_val = validator.evaluate(model, model_type=model_type_for_val)
    logger.info(
        f"Final: L2(u)={final_val.relative_l2_u:.4e}  "
        f"L2(v)={final_val.relative_l2_v:.4e}  "
        f"L2(p)={final_val.relative_l2_p:.4e}"
    )

    # --- Save metrics ---
    metrics = {
        "run_name": run_name,
        "experiment_type": exp_type,
        "total_training_time_sec": total_time,
        "final_step": trainer.state["step"],
        "validation": final_val.as_dict(),
    }
    if exp_type == "inverse":
        l1 = float(model.effective_lambda_1.item())
        l2 = float(model.effective_lambda_2.item())
        metrics["lambda_1_identified"] = l1
        metrics["lambda_2_identified"] = l2
        metrics["lambda_1_error_pct"] = parameter_error_pct(l1, TRUE_LAMBDA_1)
        metrics["lambda_2_error_pct"] = parameter_error_pct(l2, TRUE_LAMBDA_2)
        logger.info(
            f"Identified λ₁ = {l1:.6f}  (true 1.0,   error {metrics['lambda_1_error_pct']:.3f}%)"
        )
        logger.info(
            f"Identified λ₂ = {l2:.6f}  (true 0.01,  error {metrics['lambda_2_error_pct']:.3f}%)"
        )

    save_final_metrics(metrics, out_dir)
    save_history(trainer.state["history"], out_dir)

    logger.info(f"All results saved to: {out_dir}")


if __name__ == "__main__":
    main()
