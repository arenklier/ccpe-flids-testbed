"""Dataset loading + non-IID Dirichlet partitioning.

Preprocessed datasets live as parquet: a float32 feature matrix plus an int64
label column ``y``. Partitions are precomputed once (see experiments/prepare_data
.py) and written as ``client_XX.parquet`` so every run over the same (dataset,
alpha, seed) sees identical shards — no leakage between the held-out test split
and any client shard.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_parquet(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_parquet(path)
    y = torch.tensor(df["y"].to_numpy(), dtype=torch.long)
    X = torch.tensor(df.drop(columns=["y"]).to_numpy(dtype=np.float32))
    return X, y


def make_loader(path: str | Path, batch_size: int, shuffle: bool) -> DataLoader:
    X, y = load_parquet(path)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


def dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float,
                        rng: np.random.Generator) -> list[np.ndarray]:
    """Split sample indices into ``n_clients`` shards with label skew alpha.

    alpha -> 0 : each client sees few classes (extreme non-IID)
    alpha large: near-uniform (IID). Pass alpha=inf for exact IID.
    """
    n_classes = int(labels.max()) + 1
    idx_by_class = [np.where(labels == c)[0] for c in range(n_classes)]
    shards: list[list[int]] = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx = idx_by_class[c]
        rng.shuffle(idx)
        if np.isinf(alpha):
            props = np.full(n_clients, 1.0 / n_clients)
        else:
            props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx, cuts)):
            shards[cid].extend(part.tolist())

    return [np.array(sorted(s), dtype=np.int64) for s in shards]
