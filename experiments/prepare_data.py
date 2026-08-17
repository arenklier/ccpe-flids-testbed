"""Preprocess a dataset and write client shards + a held-out test parquet.

Usage:
  python -m experiments.prepare_data --dataset nbaiot --raw /datasets/nbaiot \
      --out /datasets/prepared/nbaiot --n-clients 12 --alpha 0.5 --seed 0

Outputs under <out>/<alpha>_<seed>/:
  client_00.parquet ... client_NN.parquet   (training shards, non-IID)
  test.parquet                              (global held-out test)
  meta.json                                 (in_dim, n_classes, class counts)

Leak-free protocol: the test split is carved out BEFORE partitioning, so no
test row can appear in any client shard. Features are standardized with stats
fit on the training portion only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flids.data import dirichlet_partition  # noqa: E402


def _nbaiot_label(csv: str) -> str:
    """Label from path: <device>/benign_traffic.csv, <device>/<family>/<attack>.csv."""
    parts = csv.replace("\\", "/").split("/")
    fname = parts[-1][:-4]
    if fname == "benign_traffic":
        return "benign"
    family = parts[-2]  # 'gafgyt' or 'mirai'
    return f"{family}_{fname}"


def load_nbaiot(raw: str, cap_per_file: int, rng) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pool all 9 devices; 11-class label (benign + 5 gafgyt + 5 mirai).

    Feature columns are the 115 N-BaIoT statistical features, identical across
    devices. Per-file cap keeps memory bounded; natural class imbalance is
    preserved (that imbalance is central to the macro-F1 story).
    """
    csvs = sorted(glob.glob(os.path.join(raw, "*", "benign_traffic.csv"))
                  + glob.glob(os.path.join(raw, "*", "*", "*.csv")))
    rows, labels = [], []
    classes: dict[str, int] = {}
    for csv in csvs:
        label = _nbaiot_label(csv)
        df = pd.read_csv(csv)
        if len(df) > cap_per_file:
            df = df.iloc[rng.choice(len(df), cap_per_file, replace=False)]
        classes.setdefault(label, len(classes))
        rows.append(df.to_numpy(dtype=np.float32))
        labels.append(np.full(len(df), classes[label], dtype=np.int64))
    # stable class ordering: benign first, then sorted attack names
    order = sorted(classes, key=lambda k: (k != "benign", k))
    remap = {classes[name]: i for i, name in enumerate(order)}
    X = np.concatenate(rows)
    y = np.array([remap[v] for v in np.concatenate(labels)], dtype=np.int64)
    return X, y, order


CICIDS_DROP = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
               "Protocol", "Timestamp"]


def load_cicids2017(raw: str, min_class: int, rng) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """DistriNet cleaned CICIDS2017 (Engelen et al. 2021 corrected labels).

    Pools the 5 working-day CSVs. Drops flow identifiers (IPs, ports, protocol,
    timestamp, flow id) that would leak — keeps the 76 statistical flow features.
    Merges '<attack> - Attempted' into its base class, prunes classes with fewer
    than ``min_class`` samples (extreme rarities like Heartbleed/SQLi), and
    scrubs inf/nan produced by zero-duration flows.
    """
    frames = []
    for csv in sorted(glob.glob(os.path.join(raw, "*.csv"))):
        df = pd.read_csv(csv, low_memory=False)
        df = df.drop(columns=[c for c in CICIDS_DROP if c in df.columns])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    labels = (df["Label"].astype(str).str.strip()
              .str.replace(" - Attempted", "", regex=False))
    df = df.drop(columns=["Label"])

    # numeric features only; scrub inf/nan
    X = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    X[~np.isfinite(X)] = 0.0

    # prune rare classes on the full pool
    lab = labels.to_numpy()
    keep_names = [c for c, n in zip(*np.unique(lab, return_counts=True)) if n >= min_class]
    mask = np.isin(lab, keep_names)
    X, lab = X[mask], lab[mask]

    order = sorted(set(keep_names), key=lambda k: (k != "BENIGN", k))
    idmap = {name: i for i, name in enumerate(order)}
    y = np.array([idmap[v] for v in lab], dtype=np.int64)
    return X.astype(np.float32), y, order


BOTIOT_DROP = [  # identifiers, timestamps, string dups (keep *_number), labels
    "pkSeqID", "stime", "ltime", "saddr", "daddr", "sport", "dport", "seq",
    "flgs", "proto", "state", "attack", "category", "subcategory",
]


def load_botiot(raw: str, min_class: int, rng) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """UNSW Bot-IoT 5% subset; label by ``category`` (DDoS/DoS/Recon/Normal/Theft).

    Drops raw identifiers (IPs, ports), timestamps and the string columns that
    duplicate a *_number encoding, keeping the numeric Argus flow features.
    Prunes categories below ``min_class`` (Theft, ~79 rows). The extreme
    Normal/attack imbalance is intrinsic to Bot-IoT and kept intentionally.
    """
    frames = []
    for csv in sorted(glob.glob(os.path.join(raw, "reduced_data_*.csv"))):
        frames.append(pd.read_csv(csv, low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    labels = df["category"].astype(str).str.strip().to_numpy()
    feats = df.drop(columns=[c for c in BOTIOT_DROP if c in df.columns])
    X = feats.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    X[~np.isfinite(X)] = 0.0

    keep = [c for c, n in zip(*np.unique(labels, return_counts=True)) if n >= min_class]
    mask = np.isin(labels, keep)
    X, labels = X[mask], labels[mask]
    order = sorted(set(keep), key=lambda k: (k != "Normal", k))
    idmap = {name: i for i, name in enumerate(order)}
    y = np.array([idmap[v] for v in labels], dtype=np.int64)
    return X.astype(np.float32), y, order


def make_synthetic(rng, n=60000, in_dim=64, n_classes=8):
    """Imbalanced Gaussian-blob classification for testbed validation."""
    counts = (np.geomspace(12000, 400, n_classes)).astype(int)
    centers = rng.normal(0, 3, size=(n_classes, in_dim))
    Xs, ys = [], []
    for c, cnt in enumerate(counts):
        Xs.append(rng.normal(centers[c], 1.0, size=(cnt, in_dim)).astype(np.float32))
        ys.append(np.full(cnt, c, dtype=np.int64))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    return X, y, [f"class_{i}" for i in range(n_classes)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nbaiot", "synthetic",
                                                         "cicids2017", "botiot"])
    ap.add_argument("--raw", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clients", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.5)  # use 'inf' via --iid
    ap.add_argument("--iid", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--cap-per-file", type=int, default=20000)
    ap.add_argument("--min-class", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.dataset == "nbaiot":
        X, y, names = load_nbaiot(args.raw, args.cap_per_file, rng)
    elif args.dataset == "cicids2017":
        X, y, names = load_cicids2017(args.raw, args.min_class, rng)
    elif args.dataset == "botiot":
        X, y, names = load_botiot(args.raw, args.min_class, rng)
    elif args.dataset == "synthetic":
        X, y, names = make_synthetic(rng)
    else:
        raise SystemExit(f"{args.dataset}: loader not wired yet (awaiting raw data)")

    # shuffle, carve held-out test BEFORE partitioning (leak-free)
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
    n_test = int(len(y) * args.test_frac)
    Xte, yte = X[:n_test], y[:n_test]
    Xtr, ytr = X[n_test:], y[n_test:]

    # standardize on train stats only; accumulate in float64 (N-BaIoT features
    # have large magnitudes that overflow float32 variance), then cast back
    mu = Xtr.astype(np.float64).mean(0, keepdims=True)
    sd = Xtr.astype(np.float64).std(0, keepdims=True) + 1e-8
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xte = ((Xte - mu) / sd).astype(np.float32)
    np.clip(Xtr, -10, 10, out=Xtr); np.clip(Xte, -10, 10, out=Xte)

    alpha = np.inf if args.iid else args.alpha
    shards = dirichlet_partition(ytr, args.n_clients, alpha, rng)

    tag = f"iid_{args.seed}" if args.iid else f"a{args.alpha}_{args.seed}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    cols = [f"f{i}" for i in range(X.shape[1])]

    def write(path, Xa, ya):
        df = pd.DataFrame(Xa, columns=cols); df["y"] = ya
        df.to_parquet(path, index=False)

    write(out / "test.parquet", Xte, yte)
    for cid, idx in enumerate(shards):
        write(out / f"client_{cid:02d}.parquet", Xtr[idx], ytr[idx])

    meta = {"dataset": args.dataset, "in_dim": int(X.shape[1]),
            "n_classes": int(y.max() + 1), "class_names": names,
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "alpha": (None if args.iid else args.alpha), "seed": args.seed,
            "shard_sizes": [int(len(s)) for s in shards],
            "train_class_counts": np.bincount(ytr).tolist()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prepare] {args.dataset} {tag}: in_dim={meta['in_dim']} "
          f"classes={meta['n_classes']} shards={meta['shard_sizes']}")


if __name__ == "__main__":
    main()
