"""Local stand-ins for the three ``timm`` helpers the CMX code uses.

The upstream CMX repository imports ``DropPath``, ``to_2tuple`` and
``trunc_normal_`` from ``timm.models.layers``; the project environment has
no ``timm`` installed and adding a dependency is out of scope. These are
faithful reimplementations of the three functions (timm is Apache-2.0;
the algorithms are reproduced here so behavior is identical):

- ``to_2tuple``    - timm.models.layers.helpers.to_2tuple
- ``trunc_normal_`` - timm.models.layers.weight_init._no_grad_trunc_normal_
- ``DropPath``     - timm.models.layers.drop.DropPath / drop_path

Behavior is pinned by tests in ``tests/material_fusion/test_cmx_port.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def to_2tuple(x):
    """Return ``(x, x)`` for scalars, the input unchanged for sequences."""
    if isinstance(x, (tuple, list)):
        return x
    return (x, x)


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """Truncated normal initialization, in place (timm semantics)."""
    with torch.no_grad():
        # Values are drawn from a uniform CDF interval and mapped through
        # erfinv, which is how timm avoids rejection sampling.
        l = (1.0 + math.erf((a - mean) / std / math.sqrt(2.0))) / 2.0
        u = (1.0 + math.erf((b - mean) / std / math.sqrt(2.0))) / 2.0
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth per sample (timm semantics)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Per-sample stochastic depth, identity outside training (timm semantics)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"
