"""Target-threshold sweep analysis.

The sweep runs each strategy to a high (unreachable) target so it plateaus, and
records the full macro-F1-vs-wall-clock curve. Here we post-hoc compute, for a
range of target thresholds, (a) how many seeds reach the threshold and (b) the
mean wall-clock to reach it. As the threshold rises toward the achievable
ceiling, laggard strategies stop reaching while the best strategy still does ---
this is where a method difference becomes visible.

Run after the sweep completes:
  python -m experiments.sweep_analyze --results /mnt/data/ccpe-flids/results/sweep
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

THRESHOLDS = [0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
STRAT_ORDER = ["sync", "fedbuff", "fedasync", "staleness"]


def load(results):
    runs = defaultdict(list)  # (noniid, strat) -> list of histories
    for mp in sorted(Path(results).glob("*/metrics.json")):
        d = json.loads(mp.read_text())
        c = d["config"]
        noniid = c["part_tag"].rsplit("_", 1)[0]
        runs[(noniid, c["strategy"])].append(d["history"])
    return runs


def t2t(history, thr):
    hit = next((h for h in history if h["macro_f1"] >= thr), None)
    return hit["wall_s"] if hit else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/mnt/data/ccpe-flids/results/sweep")
    args = ap.parse_args()
    runs = load(args.results)
    noniids = sorted({k[0] for k in runs})

    for ni in noniids:
        print(f"\n===== non-IID {ni}: mean wall-clock to threshold "
              f"(reached/total seeds) =====")
        header = "thr   " + "".join(f"{s:>18}" for s in STRAT_ORDER)
        print(header)
        for thr in THRESHOLDS:
            row = f"{thr:.2f}  "
            for s in STRAT_ORDER:
                hists = runs.get((ni, s), [])
                times = [t2t(h, thr) for h in hists]
                reached = [t for t in times if t is not None]
                if reached:
                    cell = f"{mean(reached):.0f}s ({len(reached)}/{len(hists)})"
                else:
                    cell = f"-- (0/{len(hists)})"
                row += f"{cell:>18}"
            print(row)
        # plateau (best f1 mean)
        plateau = {s: mean([max(h[-1]["macro_f1"] if h else 0,
                                 max((x["macro_f1"] for x in h), default=0))
                            for h in runs.get((ni, s), [[]])]) for s in STRAT_ORDER}
        print("ceiling(best-F1): " + "  ".join(f"{s}={plateau[s]:.3f}"
                                               for s in STRAT_ORDER))


if __name__ == "__main__":
    main()
