"""Sweep driver: build the experiment matrix and run each cell sequentially.

Cells are skipped if their metrics.json already exists (resumable). Use
--dry-run to print the matrix and count without launching anything. Runs MUST
be sequential — wall-clock-to-target is the primary metric, so any CPU
contention from a parallel run would corrupt it.

Matrix (full-factorial; network is a core axis because the async advantage only
appears under network heterogeneity — see the crossover result)
------
core        : 4 strategies x {homo,hetero} x {lan,mixed} x {a0.1,a0.5,iid}
              x 3 datasets x 3 seeds = 432
compression : {int8,topk} x 4 strategies x {lan,mixed} x a0.5
              x 3 datasets x 3 seeds = 144
Total 576. Mixed-network cells need SUDO_PASS in the environment (tc netem).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

STRATEGIES = ["sync", "fedasync", "fedbuff", "staleness"]
DATASETS = ["cicids2017", "nbaiot", "botiot"]
SEEDS = [0, 1, 2]
NONIID = ["a0.1", "a0.5", "iid"]
RESULTS = "/mnt/data/ccpe-flids/results/grid"

# per-dataset target macro-F1, set at ~85% of the observed stable ceiling so
# every strategy can reach it and time-to-target is comparable; best_f1 is
# reported separately for final-quality differences. Runs stop early when hit.
TARGET = {"cicids2017": 0.60, "nbaiot": 0.68, "botiot": 0.62}


def cells(kind: str):
    if kind == "core":
        for ds in DATASETS:
            for strat in STRATEGIES:
                for hw in ["homo", "hetero"]:
                    for net in ["lan", "mixed"]:
                        for noniid in NONIID:
                            for seed in SEEDS:
                                yield dict(ds=ds, strat=strat, hw=hw, noniid=noniid,
                                           seed=seed, net=net, comp="none")
    elif kind == "compression":
        for ds in DATASETS:
            for strat in STRATEGIES:
                for net in ["lan", "mixed"]:
                    for comp in ["int8", "topk"]:
                        for seed in SEEDS:
                            yield dict(ds=ds, strat=strat, hw="hetero",
                                       noniid="a0.5", seed=seed, net=net, comp=comp)


def exp_id(c: dict) -> str:
    return f"{c['ds']}__{c['strat']}__{c['hw']}__{c['noniid']}__{c['net']}__{c['comp']}__s{c['seed']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=["core", "compression"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--max-seconds", type=int, default=600)
    ap.add_argument("--max-version", type=int, default=400)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    matrix = [c for k in args.kinds for c in cells(k)
              if c["ds"] in args.datasets and c["seed"] in args.seeds
              and c["strat"] in args.strategies]
    todo = [c for c in matrix
            if not Path(f"{RESULTS}/{exp_id(c)}/metrics.json").exists()]
    print(f"matrix={len(matrix)} cells, {len(matrix)-len(todo)} done, {len(todo)} to run")
    if args.dry_run:
        for c in todo[:20]:
            print("  ", exp_id(c))
        if len(todo) > 20:
            print(f"   ... (+{len(todo)-20} more)")
        return

    for i, c in enumerate(todo, 1):
        part_tag = f"{c['noniid']}_{c['seed']}"
        eid = exp_id(c)
        print(f"\n===== [{i}/{len(todo)}] {eid} =====", flush=True)
        cmd = ["python3", "-m", "experiments.run_experiment",
               "--exp-id", f"grid/{eid}", "--dataset", c["ds"],
               "--part-tag", part_tag, "--strategy", c["strat"],
               "--hardware", c["hw"], "--network", c["net"],
               "--compression", c["comp"],
               "--target-f1", str(TARGET[c["ds"]]),
               "--max-seconds", str(args.max_seconds),
               "--max-version", str(args.max_version)]
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
