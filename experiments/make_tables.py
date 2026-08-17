"""Emit LaTeX booktabs tables from the grouped grid summary.

Run after aggregate.py:
  python -m experiments.make_tables --summary /mnt/data/ccpe-flids/results/summary_grouped.csv \
      --out /mnt/data/ccpe-flids/results/tables

Tables
  tab_crossover.tex   per dataset: strategy x {LAN,mixed} -> time-to-target
                      (mean+/-std) and best-F1 -- the headline table
  tab_compression.tex per strategy: none/int8/topk -> uplink MB and best-F1
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

STRAT_ORDER = ["sync", "fedbuff", "fedasync", "staleness"]
STRAT_LABEL = {"sync": "Sync FedAvg", "fedbuff": "FedBuff",
               "fedasync": "FedAsync", "staleness": "\\textbf{Staleness (ours)}"}
DS_LABEL = {"cicids2017": "CICIDS2017", "nbaiot": "N-BaIoT", "botiot": "Bot-IoT"}


def load(summary: str) -> list[dict]:
    with open(summary) as f:
        return list(csv.DictReader(f))


def _fmt(mean, std):
    if mean in ("", None):
        return "--"
    return f"{float(mean):.0f}$\\pm${float(std):.0f}" if std not in ("", None) \
        else f"{float(mean):.0f}"


def crossover_table(rows, out):
    # key: (ds, strat, net) -> row; filter hetero, a0.5, comp none
    idx = {(r["ds"], r["strat"], r["net"]): r for r in rows
           if r["hw"] == "hetero" and r["noniid"] == "a0.5" and r["comp"] == "none"}
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Wall-clock seconds to target macro-F1 (mean$\pm$std over "
             r"3 seeds) and best macro-F1, heterogeneous hardware, $\alpha{=}0.5$. "
             r"Sync is fastest on LAN but slowest under the mixed network; the "
             r"async ranking reverses.}",
             r"\label{tab:crossover}",
             r"\begin{tabular}{ll rr r}", r"\toprule",
             r"Dataset & Strategy & LAN (s) & Mixed (s) & Best-F1 \\", r"\midrule"]
    for ds in DS_LABEL:
        for j, s in enumerate(STRAT_ORDER):
            lan = idx.get((ds, s, "lan"), {})
            mix = idx.get((ds, s, "mixed"), {})
            dscell = DS_LABEL[ds] if j == 0 else ""
            bf1 = mix.get("best_f1_mean", "") or lan.get("best_f1_mean", "")
            bf1s = f"{float(bf1):.3f}" if bf1 not in ("", None) else "--"
            lines.append(
                f"{dscell} & {STRAT_LABEL[s]} & "
                f"{_fmt(lan.get('t2t_mean',''), lan.get('t2t_std',''))} & "
                f"{_fmt(mix.get('t2t_mean',''), mix.get('t2t_std',''))} & {bf1s} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    Path(out, "tab_crossover.tex").write_text("\n".join(lines))
    print("wrote tab_crossover.tex")


def compression_table(rows, out):
    idx = {(r["ds"], r["strat"], r["comp"]): r for r in rows
           if r["hw"] == "hetero" and r["noniid"] == "a0.5" and r["net"] == "lan"}
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Communication-accuracy trade-off (LAN, hetero, "
             r"$\alpha{=}0.5$): total uplink (MB) and best macro-F1 under int8 "
             r"quantization and top-$k$ sparsification.}",
             r"\label{tab:compression}",
             r"\begin{tabular}{ll rr rr rr}", r"\toprule",
             r"& & \multicolumn{2}{c}{none} & \multicolumn{2}{c}{int8} & "
             r"\multicolumn{2}{c}{top-$k$} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Dataset & Strategy & MB & F1 & MB & F1 & MB & F1 \\", r"\midrule"]
    for ds in DS_LABEL:
        for j, s in enumerate(STRAT_ORDER):
            cells = []
            for comp in ["none", "int8", "topk"]:
                r = idx.get((ds, s, comp), {})
                mb = r.get("uplink_mean", "")
                f1 = r.get("best_f1_mean", "")
                cells.append(f"{float(mb):.0f}" if mb not in ("", None) else "--")
                cells.append(f"{float(f1):.3f}" if f1 not in ("", None) else "--")
            dscell = DS_LABEL[ds] if j == 0 else ""
            lines.append(f"{dscell} & {STRAT_LABEL[s]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    Path(out, "tab_compression.tex").write_text("\n".join(lines))
    print("wrote tab_compression.tex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="/mnt/data/ccpe-flids/results/summary_grouped.csv")
    ap.add_argument("--out", default="/mnt/data/ccpe-flids/results/tables")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rows = load(args.summary)
    crossover_table(rows, args.out)
    compression_table(rows, args.out)
    print(f"wrote tables -> {args.out}")


if __name__ == "__main__":
    main()
