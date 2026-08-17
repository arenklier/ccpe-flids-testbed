"""Prepare every (dataset x non-IID x seed) partition the grid needs.

Runs prepare_data for alpha in {0.1, 0.5} and IID, seeds {0,1,2}, per dataset.
Idempotent: skips a partition whose meta.json already exists.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATA = "/datasets"
RAW = {"cicids2017": f"{DATA}/cicids2017",
       "nbaiot": f"{DATA}/nbaiot",
       "botiot": f"{DATA}/botiot"}
ALPHAS = [("a0.1", ["--alpha", "0.1"]),
          ("a0.5", ["--alpha", "0.5"]),
          ("iid", ["--iid"])]
SEEDS = [0, 1, 2]


def main():
    datasets = sys.argv[1:] or list(RAW)
    for ds in datasets:
        for tag, aflag in ALPHAS:
            for seed in SEEDS:
                part_tag = f"{'iid' if tag=='iid' else tag}_{seed}"
                meta = Path(f"/datasets/prepared/{ds}/{part_tag}/meta.json")
                if meta.exists():
                    print(f"[skip] {ds} {part_tag}")
                    continue
                cmd = ["python", "-m", "experiments.prepare_data",
                       "--dataset", ds, "--raw", RAW[ds],
                       "--out", f"{DATA}/prepared/{ds}",
                       "--n-clients", "12", "--seed", str(seed)] + aflag
                print(f"[prep] {ds} {part_tag}")
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
