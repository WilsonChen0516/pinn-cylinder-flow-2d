"""
Autograd helpers for PINN derivative computation.

These thin wrappers around torch.autograd.grad ensure correct handling of
create_graph / retain_graph, and make physics code cleaner to read.
"""

from __future__ import annotations

import torch


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    First-order derivative d(outputs)/d(inputs).

    outputs and inputs are assumed to have the same batch dimension;
    the returned gradient has shape matching inputs.

    Equivalent to `du/dx` when outputs=u, inputs=x.
    """
    g = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,    # needed for second-order derivatives
        retain_graph=True,
        only_inputs=True,
    )[0]
    return g


def grad_scalar(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """
    Same as grad() but assumes output is a scalar-like column vector (N, 1).
    Equivalent to grad() actually, kept for semantic clarity.
    """
    return grad(output, inputs)
