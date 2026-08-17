"""Flow-level intrusion-detection MLP.

Deliberately plain: this study's contribution is system behaviour, not model
design. No hyperparameter search, no architectural novelty.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlowMLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden=(256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


SIZES = {
    "small": (256, 128),        # default throughout the paper
    "large": (2048, 1024, 512),  # ~25x more FLOPs/step; stresses the slow tier
}


def build_model(in_dim: int, n_classes: int, size: str = "small") -> FlowMLP:
    return FlowMLP(in_dim, n_classes, hidden=SIZES[size])
