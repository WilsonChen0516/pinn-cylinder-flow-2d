"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def relative_l2(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    """Relative L2 norm error: ||pred - true||_2 / ||true||_2"""
    num = float(np.linalg.norm(pred - true))
    den = float(np.linalg.norm(true)) + eps
    return num / den


def max_abs_error(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).max())


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def parameter_error_pct(identified: float, true: float) -> float:
    return 100.0 * abs(identified - true) / (abs(true) + 1e-12)
