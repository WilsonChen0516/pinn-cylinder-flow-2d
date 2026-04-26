"""
Verify autograd-based derivatives against analytical solutions.

Run with:
    pytest tests/test_derivatives.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.derivatives import grad


# ---------- Fixtures ----------

@pytest.fixture
def xy_grid():
    """Provide (x, y) tensors with requires_grad=True on a small grid."""
    n = 200
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, n, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    y = torch.linspace(-1, 1, n, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    return x, y


# ---------- Tests ----------

def test_grad_polynomial_1st_order(xy_grid):
    """u = x^3 -> du/dx = 3x^2"""
    x, _ = xy_grid
    u = x ** 3
    u_x = grad(u, x)
    expected = 3 * x ** 2
    assert torch.allclose(u_x, expected, atol=1e-10)


def test_grad_polynomial_2nd_order(xy_grid):
    """u = x^3 -> d2u/dx2 = 6x"""
    x, _ = xy_grid
    u = x ** 3
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    expected = 6 * x
    assert torch.allclose(u_xx, expected, atol=1e-10)


def test_grad_sine(xy_grid):
    """u = sin(x) -> du/dx = cos(x), d2u/dx2 = -sin(x)"""
    x, _ = xy_grid
    u = torch.sin(x)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    assert torch.allclose(u_x, torch.cos(x), atol=1e-10)
    assert torch.allclose(u_xx, -torch.sin(x), atol=1e-10)


def test_grad_mixed_2d(xy_grid):
    """u(x,y) = x^2 * y^3 -> u_x = 2xy^3, u_y = 3x^2 y^2, u_xx = 2y^3, u_yy = 6x^2 y"""
    x, y = xy_grid
    u = (x ** 2) * (y ** 3)
    u_x = grad(u, x)
    u_y = grad(u, y)
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)

    assert torch.allclose(u_x, 2 * x * y ** 3, atol=1e-10)
    assert torch.allclose(u_y, 3 * (x ** 2) * (y ** 2), atol=1e-10)
    assert torch.allclose(u_xx, 2 * y ** 3, atol=1e-10)
    assert torch.allclose(u_yy, 6 * (x ** 2) * y, atol=1e-10)


def test_grad_through_network():
    """Check autograd still works through a small MLP."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 1),
    )

    x = torch.randn(50, 1, requires_grad=True)
    y = torch.randn(50, 1, requires_grad=True)
    inp = torch.cat([x, y], dim=1)
    u = net(inp)

    u_x = grad(u, x)
    u_y = grad(u, y)
    u_xx = grad(u_x, x)

    assert u_x.shape == x.shape
    assert u_y.shape == y.shape
    assert u_xx.shape == x.shape
    assert torch.isfinite(u_xx).all()
