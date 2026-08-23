"""Rewrite an existing partition with a controlled shard-size ratio.

The label skew of the source partition is left alone: each client's rows are
subsampled uniformly at random, which preserves its class mix in expectation.
Only the *sizes* change, and the total number of training rows is held fixed
across ratios so that a slower run cannot be blamed on having less data.

    python reskew.py --src datasets/prepared/cicids2017/a0.5_0 \
                     --out datasets/prepared/cicids2017_skew25/a0.5_0 \
                     --ratio 25 --seed 0
"""
import argparse, json, shutil
from pathlib import Path

import numpy as np
import pandas as pd


def target_sizes(src_sizes, ratio, total):
    """Geometric ladder with the requested max/min, summing to ``total``.

    The largest target is given to the largest source shard, so the targets are
    satisfiable whenever the ladder fits under the sorted source sizes.
    """
    n = len(src_sizes)
    ladder = np.array([ratio ** (i / (n - 1)) for i in range(n)], dtype=np.float64)
    ladder *= total / ladder.sum()
    order = np.argsort(src_sizes)          # smallest source shard first
    out = np.zeros(n, dtype=np.int64)
    for rank, cid in enumerate(order):
        out[cid] = int(round(ladder[rank]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--total", type=int, default=0,
                    help="rows to keep in total; default 12x the smallest shard")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    meta = json.loads((src / "meta.json").read_text())
    files = sorted(src.glob("client_*.parquet"))
    sizes = np.array([len(pd.read_parquet(f, columns=["y"])) for f in files])

    total = a.total or int(sizes.min()) * len(files)
    tgt = target_sizes(sizes, a.ratio, total)

    # a target may exceed what its client actually holds; cap and redistribute
    # the shortfall onto clients that still have room, largest first
    over = np.maximum(tgt - sizes, 0)
    if over.any():
        tgt = np.minimum(tgt, sizes)
        room = sizes - tgt
        short = int(over.sum())
        for cid in np.argsort(-room):
            if short <= 0:
                break
            take = min(short, int(room[cid]))
            tgt[cid] += take
            short -= take

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    kept = []
    for f, want in zip(files, tgt):
        df = pd.read_parquet(f)
        want = int(min(want, len(df)))
        idx = rng.choice(len(df), size=want, replace=False)
        idx.sort()
        df.iloc[idx].reset_index(drop=True).to_parquet(out / f.name, index=False)
        kept.append(want)

    shutil.copy(src / "test.parquet", out / "test.parquet")
    meta["shard_sizes"] = kept
    meta["n_train"] = int(sum(kept))
    meta["size_ratio_requested"] = a.ratio
    meta["size_ratio_actual"] = round(max(kept) / min(kept), 2)
    meta["reskewed_from"] = str(src)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("[reskew] %s ratio=%.4g total=%d shards=%s"
          % (out, meta["size_ratio_actual"], sum(kept), kept))


if __name__ == "__main__":
    main()
