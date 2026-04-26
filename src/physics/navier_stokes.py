"""
Navier-Stokes residual computation for 2D incompressible flow.

Uses the stream function formulation:
    u = d(psi)/dy
    v = -d(psi)/dx

which automatically satisfies the continuity equation (u_x + v_y = 0),
so we only enforce the two momentum equations as PDE residuals.

Parametrized form (for inverse problem compatibility):
    f_u := u_t + lambda_1 * (u*u_x + v*u_y) + p_x - lambda_2 * (u_xx + u_yy)
    f_v := v_t + lambda_1 * (u*v_x + v*v_y) + p_y - lambda_2 * (v_xx + v_yy)

With lambda_1 = 1, lambda_2 = 1/Re.
For Re=100, lambda_2 = 0.01 (ground truth).
"""

from __future__ import annotations

from typing import Tuple

import torch

from src.physics.derivatives import grad


def velocity_from_streamfunction(
    psi: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute u, v from stream function psi.
        u = d(psi)/dy
        v = -d(psi)/dx
    """
    u = grad(psi, y)
    v = -grad(psi, x)
    return u, v


def ns_residual_2d(
    psi: torch.Tensor,
    p: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    lambda_1: torch.Tensor | float,
    lambda_2: torch.Tensor | float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute 2D incompressible Navier-Stokes momentum residuals
    using stream function formulation.

    Parameters
    ----------
    psi, p : (M, 1) tensor  network outputs
    x, y, t : (M, 1) tensors with requires_grad=True
    lambda_1, lambda_2 : scalar tensors or floats
                         lambda_1 multiplies convective term (= 1 physically)
                         lambda_2 multiplies viscous term   (= 1/Re)

    Returns
    -------
    u, v   : recovered velocities (needed for data loss)
    f_u    : x-momentum residual,  shape (M, 1)
    f_v    : y-momentum residual,  shape (M, 1)
    """
    # Velocities from stream function
    u = grad(psi, y)
    v = -grad(psi, x)

    # First-order velocity derivatives
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)
    v_t = grad(v, t)
    v_x = grad(v, x)
    v_y = grad(v, y)

    # Second-order velocity derivatives
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)

    # Pressure gradients
    p_x = grad(p, x)
    p_y = grad(p, y)

    # Momentum residuals
    f_u = u_t + lambda_1 * (u * u_x + v * u_y) + p_x - lambda_2 * (u_xx + u_yy)
    f_v = v_t + lambda_1 * (u * v_x + v * v_y) + p_y - lambda_2 * (v_xx + v_yy)

    return u, v, f_u, f_v
