"""
Data loader for Raissi et al. (2019) cylinder wake ground truth.

The .mat file contains:
    U_star : (N, 2, T)  velocity, index 0 = u, 1 = v
    p_star : (N, T)     pressure
    t      : (T, 1)     time stamps
    X_star : (N, 2)     spatial coordinates, index 0 = x, 1 = y

Where N = 5000 spatial points, T = 200 time steps.
Spatial domain: x in [1, 8], y in [-2, 2]
Time domain:    t in [0, 19.9], dt = 0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as sio


@dataclass
class Domain:
    """Spatial-temporal bounds."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float

    def as_lower_upper(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lb, ub) arrays of shape (3,) for [x, y, t]."""
        lb = np.array([self.x_min, self.y_min, self.t_min], dtype=np.float32)
        ub = np.array([self.x_max, self.y_max, self.t_max], dtype=np.float32)
        return lb, ub


class CylinderWakeDataset:
    """
    Ground-truth dataset from Raissi's Nektar++ reference solution.

    Holds the full spatio-temporal field as flat arrays of length N*T,
    where each entry is one (x, y, t) sample with associated (u, v, p).

    Attributes
    ----------
    x, y, t : np.ndarray, shape (N*T, 1)
    u, v, p : np.ndarray, shape (N*T, 1)
    N : int   number of spatial points
    T : int   number of time steps
    domain : Domain
    """

    def __init__(self, mat_path: str | Path):
        mat_path = Path(mat_path)
        if not mat_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {mat_path}\n"
                f"Run: python scripts/download_data.py"
            )

        data = sio.loadmat(str(mat_path))

        # Raw arrays
        U_star = data["U_star"]  # (N, 2, T)
        p_star = data["p_star"]  # (N, T)
        t_raw = data["t"].flatten()  # (T,)
        X_raw = data["X_star"]  # (N, 2)

        self.N, self.T = p_star.shape
        assert U_star.shape == (self.N, 2, self.T), f"Unexpected U shape: {U_star.shape}"
        assert X_raw.shape == (self.N, 2), f"Unexpected X shape: {X_raw.shape}"

        # Flatten to (N*T, 1): every spatial point at every time
        # Use tiling convention: for a given time t_k, all N points, then next t_k
        XX = np.tile(X_raw[:, 0:1], (1, self.T))  # (N, T)
        YY = np.tile(X_raw[:, 1:2], (1, self.T))  # (N, T)
        TT = np.tile(t_raw[None, :], (self.N, 1))  # (N, T)

        self.x = XX.flatten()[:, None].astype(np.float32)  # (N*T, 1)
        self.y = YY.flatten()[:, None].astype(np.float32)
        self.t = TT.flatten()[:, None].astype(np.float32)
        self.u = U_star[:, 0, :].flatten()[:, None].astype(np.float32)
        self.v = U_star[:, 1, :].flatten()[:, None].astype(np.float32)
        self.p = p_star.flatten()[:, None].astype(np.float32)

        # Keep structured views too (useful for plotting)
        self.X_star = X_raw.astype(np.float32)  # (N, 2)
        self.t_star = t_raw.astype(np.float32)  # (T,)
        self.U_star = U_star.astype(np.float32)  # (N, 2, T)
        self.p_star = p_star.astype(np.float32)  # (N, T)

        self.domain = Domain(
            x_min=float(self.x.min()),
            x_max=float(self.x.max()),
            y_min=float(self.y.min()),
            y_max=float(self.y.max()),
            t_min=float(self.t.min()),
            t_max=float(self.t.max()),
        )

    # ----- Convenience accessors -----

    def total_points(self) -> int:
        return self.N * self.T

    def snapshot(self, time_index: int) -> dict:
        """Return u, v, p fields at a single time step, along with coords."""
        if not (0 <= time_index < self.T):
            raise IndexError(f"time_index {time_index} out of range [0, {self.T})")
        return {
            "x": self.X_star[:, 0],  # (N,)
            "y": self.X_star[:, 1],
            "u": self.U_star[:, 0, time_index],
            "v": self.U_star[:, 1, time_index],
            "p": self.p_star[:, time_index],
            "t": float(self.t_star[time_index]),
        }

    def random_subsample(
        self, n_points: int, seed: int = 0
    ) -> dict:
        """
        Randomly pick n_points from the full (N*T) spatio-temporal cloud.
        Used for inverse-problem observation sets.

        Returns
        -------
        dict with keys x, y, t, u, v, p, each of shape (n_points, 1)
        """
        total = self.total_points()
        if n_points > total:
            raise ValueError(f"Requested {n_points} > available {total}")

        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=n_points, replace=False)
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "t": self.t[idx],
            "u": self.u[idx],
            "v": self.v[idx],
            "p": self.p[idx],
            "indices": idx,
        }

    # ----- Summary -----

    def summary(self) -> str:
        d = self.domain
        return (
            f"CylinderWakeDataset\n"
            f"  N (spatial) : {self.N}\n"
            f"  T (time)    : {self.T}\n"
            f"  Total pts   : {self.total_points()}\n"
            f"  x range     : [{d.x_min:.3f}, {d.x_max:.3f}]\n"
            f"  y range     : [{d.y_min:.3f}, {d.y_max:.3f}]\n"
            f"  t range     : [{d.t_min:.3f}, {d.t_max:.3f}]\n"
            f"  u range     : [{self.u.min():.3f}, {self.u.max():.3f}]\n"
            f"  v range     : [{self.v.min():.3f}, {self.v.max():.3f}]\n"
            f"  p range     : [{self.p.min():.3f}, {self.p.max():.3f}]"
        )

    def __repr__(self) -> str:
        return self.summary()


# ----- Normalization helper -----

def normalize(
    coords: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> np.ndarray:
    """
    Map coords from [lb, ub] to [-1, 1].

    This is CRITICAL for PINN convergence — tanh saturates on large inputs.

    Parameters
    ----------
    coords : (M, D) array
    lb, ub : (D,) arrays
    """
    return 2.0 * (coords - lb) / (ub - lb) - 1.0


def denormalize(
    coords_norm: np.ndarray, lb: np.ndarray, ub: np.ndarray
) -> np.ndarray:
    """Inverse of `normalize`."""
    return (coords_norm + 1.0) * (ub - lb) / 2.0 + lb
