"""Shared param<->numpy plumbing and byte-accounting."""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


def get_params(model: nn.Module) -> List[np.ndarray]:
    """State dict as an ordered list of float32 arrays (deterministic order)."""
    return [p.detach().cpu().numpy().astype(np.float32)
            for _, p in sorted(model.state_dict().items())]


def set_params(model: nn.Module, arrays: List[np.ndarray]) -> None:
    keys = sorted(model.state_dict().keys())
    sd = {k: torch.tensor(a) for k, a in zip(keys, arrays)}
    model.load_state_dict(sd, strict=True)


def param_shapes(model: nn.Module) -> list[tuple]:
    return [tuple(p.shape) for _, p in sorted(model.state_dict().items())]


def sub(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    return [x - y for x, y in zip(a, b)]


def add(a: List[np.ndarray], b: List[np.ndarray], scale: float = 1.0) -> List[np.ndarray]:
    return [x + scale * y for x, y in zip(a, b)]
