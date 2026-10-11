"""
Untitled design — written by NEURAX from the design it analysed.

Every block below is built from the values NEURAX resolved for it, so
this model has the 43393 parameters its analysis reports. Regenerate it after any
change on the canvas.
"""
import math
from contextvars import ContextVar
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph(NamedTuple):
    """A batch of graphs: node features, edges as `[2, E]` indices, and the
    graph each node belongs to."""

    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    num_graphs: int
    # Present when the data states them: edge features, edge and node types.
    edge_attr: Optional[torch.Tensor] = None
    edge_type: Optional[torch.Tensor] = None
    node_type: Optional[torch.Tensor] = None


ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "softplus": nn.Softplus,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def activation(name):
    """The activation a block names; `none` is the identity."""
    if name in (None, "none", "identity", "linear"):
        return nn.Identity()
    if name == "sin":
        return Sin()
    return ACTIVATIONS[name]()


def features(value):
    """The tensor a block computes on: a graph's node features, else itself."""
    return value.x if isinstance(value, Graph) else value


def with_features(value, x):
    """`value` carrying new features: a graph keeps its edges."""
    return value._replace(x=x) if isinstance(value, Graph) else x


def per_position(value, f):
    """`f` applied to each position's vector: over the last axis, or over the
    channels of each pixel of a feature map — the compiler's reading of a
    linear map or a norm on an image."""
    x = features(value)
    if x.dim() == 4:
        return with_features(value, f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
    return with_features(value, f(x))


class Identity(nn.Module):
    """A block that passes what it receives on: the design's input and output,
    a scheduler, a position scheme applied elsewhere."""

    def forward(self, x, *rest):
        return x


class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return with_features(x, self.drop(features(x)))


class Dense(nn.Module):
    """A linear map over the last axis, then its activation."""

    def __init__(self, in_features, out_features, bias, function):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.function = activation(function)

    def forward(self, x):
        return per_position(x, lambda h: self.function(self.linear(h)))


class UntitledDesign(nn.Module):
    def __init__(self):
        super().__init__()
        shared = {}
        self.b_input_1 = Identity()
        self.b_proj_1 = Dense(in_features=273, out_features=128, bias=True, function="relu")
        self.b_dropout_1 = Dropout(p=0.1)
        self.b_proj_2 = Dense(in_features=128, out_features=64, bias=True, function="relu")
        self.b_dropout_2 = Dropout(p=0.1)
        self.b_proj_3 = Dense(in_features=64, out_features=1, bias=True, function=None)
        self.b_output_1 = Identity()

    def forward(self, x):
        t_input_1 = x
        t_proj_1 = self.b_proj_1(t_input_1)
        t_dropout_1 = self.b_dropout_1(t_proj_1)
        t_proj_2 = self.b_proj_2(t_dropout_1)
        t_dropout_2 = self.b_dropout_2(t_proj_2)
        t_proj_3 = self.b_proj_3(t_dropout_2)
        t_output_1 = self.b_output_1(t_proj_3)
        return t_output_1


if __name__ == "__main__":
    model = UntitledDesign()
    built = sum(p.numel() for p in model.parameters())
    print(f"UntitledDesign: {built:,} parameters")
    if built != 43393:
        print("WARNING: NEURAX's analysis counts 43393 parameters for this design")
