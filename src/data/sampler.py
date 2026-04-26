"""
Sampling utilities for PINN training.

- Latin Hypercube Sampling (LHS) for collocation points
- Boundary sampling for BC enforcement
- Initial condition sampling
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def latin_hypercube(
    n_points: int,
    lb: np.ndarray,
    ub: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """
    Latin Hypercube Sampling in D-dimensional box [lb, ub].

    Returns (n_points, D) array.

    LHS gives better space-filling than uniform sampling, which matters
    for PINN collocation where we want the PDE residual enforced evenly
    across the domain.
    """
    assert lb.shape == ub.shape, "lb, ub must have same shape"
    assert lb.ndim == 1, "lb, ub must be 1D"
    d = lb.shape[0]
    rng = np.random.default_rng(seed)

    # Stratified samples in each dimension: for each of D dims, split [0,1]
    # into n_points equal strata and pick a random point in each, then shuffle.
    # Reshape arange to (n_points, 1) so it broadcasts against (n_points, d).
    strata = np.arange(n_points, dtype=np.float64).reshape(-1, 1)  # (n_points, 1)
    jitter = rng.random((n_points, d))                              # (n_points, d)
    segments = (strata + jitter) / n_points                         # (n_points, d)

    # Independently shuffle each column (so dimensions aren't correlated)
    for j in range(d):
        rng.shuffle(segments[:, j])

    return (lb + segments * (ub - lb)).astype(np.float32)


def sample_boundary_2d_rect(
    n_per_edge: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    t_range: Tuple[float, float],
    seed: int = 0,
) -> dict:
    """
    Sample points on the boundary of a 2D rectangle x (time domain).

    Returns a dict with keys:
        'x', 'y', 't' : (4*n_per_edge*Nt, 1) arrays
        'edge'        : string labels for each point (useful for plotting)

    The four edges are: left (x=x_min), right (x=x_max),
                       bottom (y=y_min), top (y=y_max).

    For each edge, n_per_edge points are sampled in the free coordinate
    and in time, giving n_per_edge * Nt total points per edge.
    Here we sample n_per_edge points and n_per_edge times, so 4*n_per_edge^2
    points total when used straightforwardly; adjust as needed.
    """
    rng = np.random.default_rng(seed)
    x_min, x_max = x_range
    y_min, y_max = y_range
    t_min, t_max = t_range

    def random_in(low: float, high: float, n: int) -> np.ndarray:
        return rng.uniform(low, high, size=(n, 1)).astype(np.float32)

    # Left edge: x = x_min, y free, t free
    xL = np.full((n_per_edge, 1), x_min, dtype=np.float32)
    yL = random_in(y_min, y_max, n_per_edge)
    tL = random_in(t_min, t_max, n_per_edge)

    # Right edge: x = x_max
    xR = np.full((n_per_edge, 1), x_max, dtype=np.float32)
    yR = random_in(y_min, y_max, n_per_edge)
    tR = random_in(t_min, t_max, n_per_edge)

    # Bottom edge: y = y_min
    xB = random_in(x_min, x_max, n_per_edge)
    yB = np.full((n_per_edge, 1), y_min, dtype=np.float32)
    tB = random_in(t_min, t_max, n_per_edge)

    # Top edge: y = y_max
    xT = random_in(x_min, x_max, n_per_edge)
    yT = np.full((n_per_edge, 1), y_max, dtype=np.float32)
    tT = random_in(t_min, t_max, n_per_edge)

    x = np.vstack([xL, xR, xB, xT])
    y = np.vstack([yL, yR, yB, yT])
    t = np.vstack([tL, tR, tB, tT])
    edge = (
        ["left"] * n_per_edge
        + ["right"] * n_per_edge
        + ["bottom"] * n_per_edge
        + ["top"] * n_per_edge
    )

    return {"x": x, "y": y, "t": t, "edge": np.array(edge)}


def sample_initial_condition(
    n_points: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    t0: float = 0.0,
    seed: int = 0,
) -> dict:
    """Uniform random samples at a fixed initial time."""
    rng = np.random.default_rng(seed)
    x_min, x_max = x_range
    y_min, y_max = y_range

    x = rng.uniform(x_min, x_max, size=(n_points, 1)).astype(np.float32)
    y = rng.uniform(y_min, y_max, size=(n_points, 1)).astype(np.float32)
    t = np.full((n_points, 1), t0, dtype=np.float32)

    return {"x": x, "y": y, "t": t}