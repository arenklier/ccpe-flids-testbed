"""Sanity floor + centralized upper bound for every dataset.

For each prepared dataset (using the a0.5 seed-0 split's pooled clients as the
full training set and its held-out test):
  majority : always predict the majority class -> macro-F1 sanity floor
  central  : train the same MLP centrally on the pooled training data -> the
             no-federation upper bound that FL runs should approach but not beat

Run in the container (needs torch). Writes results/baselines.json.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flids.data import load_parquet  # noqa: E402
from flids.model import build_model  # noqa: E402

PREP = "/datasets/prepared"
DATASETS = ["cicids2017", "nbaiot", "botiot"]


def macro_f1(pred, y, n_classes):
    tp = np.zeros(n_classes); fp = np.zeros(n_classes); fn = np.zeros(n_classes)
    for c in range(n_classes):
        tp[c] = int(((pred == c) & (y == c)).sum())
        fp[c] = int(((pred == c) & (y != c)).sum())
        fn[c] = int(((pred != c) & (y == c)).sum())
    f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    return float(f1.mean())


def load_pooled(ds):
    tag = f"{PREP}/{ds}/a0.5_0"
    meta = json.loads(Path(tag, "meta.json").read_text())
    Xs, ys = [], []
    for cp in sorted(glob.glob(f"{tag}/client_*.parquet")):
        X, y = load_parquet(cp); Xs.append(X); ys.append(y)
    Xtr = torch.cat(Xs); ytr = torch.cat(ys)
    Xte, yte = load_parquet(f"{tag}/test.parquet")
    return Xtr, ytr, Xte, yte, meta


def central_train(Xtr, ytr, Xte, yte, n_classes, device, epochs=15):
    model = build_model(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    n = len(ytr); bs = 4096
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].to(device); yb = ytr[idx].to(device)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(yte), 8192):
            preds.append(model(Xte[i:i + 8192].to(device)).argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    return macro_f1(pred, yte.numpy(), n_classes)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for ds in DATASETS:
        Xtr, ytr, Xte, yte, meta = load_pooled(ds)
        nc = meta["n_classes"]
        maj = int(np.bincount(yte.numpy()).argmax())
        floor = macro_f1(np.full(len(yte), maj), yte.numpy(), nc)
        upper = central_train(Xtr, ytr, Xte, yte, nc, device)
        out[ds] = {"n_classes": nc, "majority_floor_macro_f1": round(floor, 4),
                   "centralized_upper_macro_f1": round(upper, 4),
                   "n_train": len(ytr), "n_test": len(yte)}
        print(f"{ds}: floor={floor:.4f} upper={upper:.4f} (device={device})", flush=True)
    Path("/mnt/data/ccpe-flids/results/baselines.json").write_text(json.dumps(out, indent=2))
    print("wrote results/baselines.json")


if __name__ == "__main__":
    main()
