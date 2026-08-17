"""Collect grid metrics.json into a tidy CSV + a grouped summary table.

Per run we extract the systems metrics the paper reports:
  time_to_target   wall-clock seconds to first reach the target macro-F1 (the
                   headline metric; NaN if never reached within the budget)
  ver_to_target    server versions to reach it
  best_f1          best macro-F1 observed
  uplink_at_target uplink MB communicated up to the moment target was hit
                   (or total uplink if never reached)
Then aggregates mean+/-std across seeds for each (dataset, strategy, hardware,
non-IID, network, compression) configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev

RESULTS = "/mnt/data/ccpe-flids/results/grid"
FIELDS = ["ds", "strat", "hw", "noniid", "net", "comp", "seed"]


def parse_id(name: str) -> dict:
    ds, strat, hw, noniid, net, comp, seed = name.split("__")
    return dict(ds=ds, strat=strat, hw=hw, noniid=noniid, net=net,
                comp=comp, seed=int(seed[1:]))


def summarize_run(d: dict) -> dict:
    hist = d["history"]
    target = d["config"]["target_macro_f1"]
    hit = next((h for h in hist if h["macro_f1"] >= target), None)
    best = max((h["macro_f1"] for h in hist), default=0.0)
    last = hist[-1] if hist else {"uplink_mb": 0.0}
    return {"time_to_target": hit["wall_s"] if hit else math.nan,
            "ver_to_target": hit["version"] if hit else math.nan,
            "best_f1": best,
            "uplink_at_target": (hit["uplink_mb"] if hit else last["uplink_mb"]),
            "reached": bool(hit)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--out", default="/mnt/data/ccpe-flids/results/summary.csv")
    args = ap.parse_args()

    rows = []
    for mp in sorted(Path(args.results).glob("*/metrics.json")):
        try:
            d = json.loads(mp.read_text())
        except Exception:
            continue
        row = parse_id(mp.parent.name)
        row.update(summarize_run(d))
        rows.append(row)

    if not rows:
        print("no runs found")
        return

    metric_cols = ["time_to_target", "ver_to_target", "best_f1",
                   "uplink_at_target", "reached"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + metric_cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} runs -> {args.out}")

    # group across seeds
    groups: dict[tuple, list] = {}
    for r in rows:
        key = tuple(r[k] for k in FIELDS if k != "seed")
        groups.setdefault(key, []).append(r)

    gkeys = [k for k in FIELDS if k != "seed"]
    summ_path = Path(args.out).with_name("summary_grouped.csv")
    with open(summ_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(gkeys + ["n", "reached_frac",
                            "t2t_mean", "t2t_std", "best_f1_mean", "best_f1_std",
                            "uplink_mean"])
        for key, rs in sorted(groups.items()):
            reached = [r for r in rs if r["reached"]]
            t2t = [r["time_to_target"] for r in reached]
            bf1 = [r["best_f1"] for r in rs]
            up = [r["uplink_at_target"] for r in rs]
            w.writerow(list(key) + [
                len(rs), round(len(reached) / len(rs), 2),
                round(mean(t2t), 1) if t2t else "",
                round(pstdev(t2t), 1) if len(t2t) > 1 else "",
                round(mean(bf1), 4), round(pstdev(bf1), 4) if len(bf1) > 1 else "",
                round(mean(up), 1)])
    print(f"wrote grouped summary -> {summ_path}")


if __name__ == "__main__":
    main()
