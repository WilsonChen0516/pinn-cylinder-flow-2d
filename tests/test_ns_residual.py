"""
Verify NS residual against Taylor-Green vortex analytical solution.

The 2D Taylor-Green vortex is an exact unsteady solution of the
incompressible Navier-Stokes equations:

    u(x, y, t) = -cos(x) sin(y) * F(t)
    v(x, y, t) =  sin(x) cos(y) * F(t)
    p(x, y, t) = -0.25 * (cos(2x) + cos(2y)) * F(t)^2

where F(t) = exp(-2 * nu * t) and nu = 1/Re.

Our stream function:  psi = cos(x) cos(y) * F(t)
Check:  u = d(psi)/dy = -cos(x) sin(y) * F(t)   ✓
        v = -d(psi)/dx = sin(x) cos(y) * F(t)   ✓

This solution satisfies NS exactly, so residual should be ~0.

Run with:
    pytest tests/test_ns_residual.py -v
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.navier_stokes import ns_residual_2d, velocity_from_streamfunction


def test_taylor_green_streamfunction_gives_correct_velocity():
    """psi = cos(x)cos(y) F(t) should yield the TG velocity field."""
    nu = 0.01
    n = 50
    torch.manual_seed(0)

    x = torch.rand(n, 1, dtype=torch.float64).requires_grad_(True)
    y = torch.rand(n, 1, dtype=torch.float64).requires_grad_(True)
    t = torch.rand(n, 1, dtype=torch.float64).requires_grad_(True)

    F = torch.exp(-2 * nu * t)
    psi = torch.cos(x) * torch.cos(y) * F

    u, v = velocity_from_streamfunction(psi, x, y)

    # Expected
    u_expected = -torch.cos(x) * torch.sin(y) * F
    v_expected = torch.sin(x) * torch.cos(y) * F

    assert torch.allclose(u, u_expected, atol=1e-10)
    assert torch.allclose(v, v_expected, atol=1e-10)


def test_taylor_green_satisfies_ns_residual():
    """
    The Taylor-Green solution satisfies NS exactly, so residual should be near zero.
    """
    nu = 0.01
    lambda_1 = 1.0
    lambda_2 = nu

    n = 100
    torch.manual_seed(42)

    # Random points in a box
    x = (2 * torch.rand(n, 1, dtype=torch.float64) - 1).requires_grad_(True)
    y = (2 * torch.rand(n, 1, dtype=torch.float64) - 1).requires_grad_(True)
    t = torch.rand(n, 1, dtype=torch.float64).requires_grad_(True)

    F = torch.exp(-2 * nu * t)
    psi = torch.cos(x) * torch.cos(y) * F
    p = -0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * F ** 2

    u, v, f_u, f_v = ns_residual_2d(
        psi=psi, p=p, x=x, y=y, t=t,
        lambda_1=lambda_1, lambda_2=lambda_2,
    )

    # Residual should be essentially zero (numerical precision)
    max_f_u = torch.abs(f_u).max().item()
    max_f_v = torch.abs(f_v).max().item()

    assert max_f_u < 1e-10, f"f_u residual too large: {max_f_u}"
    assert max_f_v < 1e-10, f"f_v residual too large: {max_f_v}"


def test_ns_residual_wrong_lambda_fails():
    """If we use wrong lambda_2, residual should NOT be zero."""
    nu = 0.01
    wrong_lambda_2 = 0.1   # wrong viscosity

    n = 50
    torch.manual_seed(0)

    x = torch.rand(n, 1, dtype=torch.float64).requires_grad_(True)
    y = torch.rand(n, 1, dtype=torch.float64).requires_grad_(True)
    t = torch.rand(n, 1, dtype=torch.float64).requires_grad_(True)

    F = torch.exp(-2 * nu * t)
    psi = torch.cos(x) * torch.cos(y) * F
    p = -0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * F ** 2

    _, _, f_u, f_v = ns_residual_2d(
        psi=psi, p=p, x=x, y=y, t=t,
        lambda_1=1.0, lambda_2=wrong_lambda_2,
    )

    # With wrong lambda, residual should be significantly non-zero
    max_residual = max(torch.abs(f_u).max().item(), torch.abs(f_v).max().item())
    assert max_residual > 1e-4, f"Expected nonzero residual with wrong lambda, got {max_residual}"
