"""
Small helpers for network construction.
"""

import torch.nn as nn


def xavier_init(module: nn.Module) -> None:
    """Apply Xavier normal initialization to Linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_layers: int,
    neurons_per_layer: int,
    activation: str = "tanh",
) -> nn.Sequential:
    """
    Build a fully-connected network with specified depth and width.

    Activation is applied after every hidden layer but not after output.
    """
    act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    act_cls = act_map[activation.lower()]

    layers = [nn.Linear(in_dim, neurons_per_layer), act_cls()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(neurons_per_layer, neurons_per_layer), act_cls()]
    layers += [nn.Linear(neurons_per_layer, out_dim)]

    net = nn.Sequential(*layers)
    net.apply(xavier_init)
    return net
