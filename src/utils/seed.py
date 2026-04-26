"""Reproducibility: set all RNG seeds."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix seeds for Python, NumPy, PyTorch (CPU and CUDA)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
