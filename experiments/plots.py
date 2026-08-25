"""Generate the paper figures from grid results.

Run after the grid finishes (needs matplotlib):
  python -m experiments.plots --results /mnt/data/ccpe-flids/results/grid \
      --out /mnt/data/ccpe-flids/results/figures

Figures
  fig_crossover_<ds>.pdf   time-to-target, lan vs mixed, per strategy (headline)
  fig_comm_<ds>.pdf        best-F1 vs uplink MB, per compression scheme
  fig_noniid_<ds>.pdf      time-to-target vs Dirichlet alpha, per strategy
  fig_curves_<ds>.pdf      macro-F1 vs wall-clock convergence, per strategy
  fig_delaycurve_cicids2017.pdf  time-to-target vs injected delay, sync vs
                                 staleness-aware (from results/fix_delaycurve)

NOTE (eval_every checkpoint-granularity correction): "time to target" (t2t) is
no longer the wall_s of the first checkpoint at/above target. Because
eval_every=5 (or =2 for the crosshost trials) counts "versions" that mean a
full barrier-gated round for sync but a single update/flush for the async
strategies, sync's checkpoints land ~3-4x further apart in wall-clock time
under the mixed network -- so the raw "first hit" reading systematically
overestimates sync's crossing time more than the async strategies'. t2t is
now computed by linearly interpolating between the last sub-target checkpoint
and the first at/above-target checkpoint.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# same order as the tables, so a reader can hold both at once
STRAT_ORDER = ["sync", "fedasync", "fedbuff", "staleness"]
STRAT_LABEL = {"sync": "Sync FedAvg", "fedbuff": "FedBuff",
               "fedasync": "FedAsync", "staleness": "Size-weighted"}
COLORS = {"sync": "#4C78A8", "fedbuff": "#F58518",
          "fedasync": "#54A24B", "staleness": "#E45756"}
DS_LABEL = {"cicids2017": "CICIDS2017", "nbaiot": "N-BaIoT",
            "botiot": "Bot-IoT"}


def ds_name(ds):
    return DS_LABEL.get(ds, ds)


def interp_t2t(history: list[dict], target: float) -> float | None:
    """Time to reach `target` macro-F1, linearly interpolated between the
    last sub-target checkpoint and the first at/above-target checkpoint
    (instead of just taking the first at/above-target checkpoint raw)."""
    raw = next((h["wall_s"] for h in history if h["macro_f1"] >= target), None)
    if raw is None:
        return None
    for i in range(1, len(history)):
        if history[i]["macro_f1"] >= target and history[i - 1]["macro_f1"] < target:
            f0, f1 = history[i - 1]["macro_f1"], history[i]["macro_f1"]
            t0, t1 = history[i - 1]["wall_s"], history[i]["wall_s"]
            frac = (target - f0) / (f1 - f0) if f1 != f0 else 0.0
            return t0 + frac * (t1 - t0)
    return raw  # target already met at the first checkpoint; can't interpolate further back


def load_runs(results: str) -> list[dict]:
    runs = []
    for mp in sorted(Path(results).glob("*/metrics.json")):
        try:
            d = json.loads(mp.read_text())
        except Exception:
            continue
        c = d["config"]
        hist = d["history"]
        tgt = c["target_macro_f1"]
        t2t = interp_t2t(hist, tgt)
        noniid, seed = c["part_tag"].rsplit("_", 1)
        runs.append({
            "dataset": c["dataset"], "strat": c["strategy"], "hw": c["hardware"],
            "net": c["network"], "comp": c["compression"], "noniid": noniid,
            "seed": int(seed), "history": hist,
            "t2t": t2t,
            "best_f1": max((h["macro_f1"] for h in hist), default=0.0),
            "uplink": hist[-1]["uplink_mb"] if hist else 0.0})
    return runs


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def fig_crossover(runs, ds, out):
    """Headline: time-to-target on LAN vs mixed network, per strategy."""
    sub = [r for r in runs if r["dataset"] == ds and r["hw"] == "hetero"
           and r["noniid"] == "a0.5" and r["comp"] == "none"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(STRAT_ORDER)); w = 0.38
    for i, net in enumerate(["lan", "mixed"]):
        vals = [_mean([r["t2t"] for r in sub if r["strat"] == s and r["net"] == net])
                for s in STRAT_ORDER]
        vals = [v if v is not None else 0 for v in vals]
        ax.bar([xi + (i - 0.5) * w for xi in x], vals, w,
               label=net.upper(), color="#BAB0AC" if net == "lan" else "#E45756")
    ax.set_xticks(list(x))
    ax.set_xticklabels([STRAT_LABEL[s].replace(" ", "\n") for s in STRAT_ORDER], fontsize=8)
    ax.set_ylabel("Wall-clock to target macro-F1 (s)")
    ax.set_title(f"{ds_name(ds)}: wall-clock to target, LAN vs mixed network")
    ax.legend(title="Network"); fig.tight_layout()
    fig.savefig(Path(out) / f"fig_crossover_{ds}.pdf")
    plt.close(fig)


def fig_comm(runs, ds, out):
    """Communication-accuracy: best-F1 vs uplink, per compression scheme."""
    sub = [r for r in runs if r["dataset"] == ds and r["hw"] == "hetero"
           and r["noniid"] == "a0.5" and r["net"] == "lan"]
    fig, ax = plt.subplots(figsize=(6, 4))
    marker = {"none": "o", "int8": "s", "topk": "^"}
    for s in STRAT_ORDER:
        for comp in ["none", "int8", "topk"]:
            rs = [r for r in sub if r["strat"] == s and r["comp"] == comp]
            if not rs:
                continue
            ax.scatter(_mean([r["uplink"] for r in rs]),
                       _mean([r["best_f1"] for r in rs]),
                       c=COLORS[s], marker=marker[comp], s=60,
                       edgecolor="k", linewidth=0.4)
    ax.set_xlabel("Total uplink communicated (MB)")
    ax.set_ylabel("Macro-F1 at the stopping checkpoint")
    ax.set_title(f"{ds_name(ds)}: communication versus accuracy")
    handles = [plt.Line2D([], [], color=COLORS[s], marker="o", ls="",
                          label=STRAT_LABEL[s]) for s in STRAT_ORDER]
    handles += [plt.Line2D([], [], color="k", marker=marker[c], ls="", label=c)
                for c in ["none", "int8", "topk"]]
    ax.legend(handles=handles, fontsize=7); fig.tight_layout()
    fig.savefig(Path(out) / f"fig_comm_{ds}.pdf")
    plt.close(fig)


def fig_noniid(runs, ds, out):
    sub = [r for r in runs if r["dataset"] == ds and r["hw"] == "hetero"
           and r["net"] == "mixed" and r["comp"] == "none"]
    order = {"a0.1": 0, "a0.5": 1, "iid": 2}
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in STRAT_ORDER:
        xs = sorted({r["noniid"] for r in sub}, key=lambda a: order[a])
        ys = [_mean([r["t2t"] for r in sub if r["strat"] == s and r["noniid"] == a])
              for a in xs]
        ax.plot([order[a] for a in xs], [y if y else float("nan") for y in ys],
                "-o", color=COLORS[s], label=STRAT_LABEL[s])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["α=0.1", "α=0.5", "IID"])
    ax.set_ylabel("Wall-clock to target (s)")
    ax.set_title(f"{ds_name(ds)}: label skew against wall-clock to target")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(Path(out) / f"fig_noniid_{ds}.pdf")
    plt.close(fig)


def fig_curves(runs, ds, out):
    sub = [r for r in runs if r["dataset"] == ds and r["hw"] == "hetero"
           and r["noniid"] == "a0.5" and r["net"] == "mixed"
           and r["comp"] == "none" and r["seed"] == 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in STRAT_ORDER:
        rs = [r for r in sub if r["strat"] == s]
        if not rs:
            continue
        h = rs[0]["history"]
        ax.plot([x["wall_s"] for x in h], [x["macro_f1"] for x in h],
                color=COLORS[s], label=STRAT_LABEL[s])
    ax.set_xlabel("Wall-clock (s)"); ax.set_ylabel("Macro-F1")
    ax.set_title(f"{ds_name(ds)}: convergence (mixed network, α=0.5)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(Path(out) / f"fig_curves_{ds}.pdf")
    plt.close(fig)


def load_delaycurve_runs(results: str) -> dict:
    """Load the continuous injected-delay sweep (results/fix_delaycurve),
    which uses a distinct <dataset>__<strategy>__d<ms>__s<seed>/metrics.json
    layout (not the grid's part_tag/network/compression layout)."""
    runs = defaultdict(list)  # (strategy, delay_ms) -> [t2t, ...]
    for mp in sorted(Path(results).glob("*/metrics.json")):
        try:
            d = json.loads(mp.read_text())
        except Exception:
            continue
        c, hist = d["config"], d["history"]
        name_parts = mp.parent.name.split("__")
        strat, dtag = name_parts[1], name_parts[2]
        delay_ms = int(dtag.lstrip("d"))
        runs[(strat, delay_ms)].append(interp_t2t(hist, c["target_macro_f1"]))
    return runs


def fig_delaycurve(delaycurve_results: str, out: str, ds: str = "cicids2017"):
    """Time-to-target vs injected network delay, sync vs staleness-aware."""
    runs = load_delaycurve_runs(delaycurve_results)
    if not runs:
        return
    delays = sorted({dl for (_, dl) in runs})
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in ("sync", "staleness"):
        ys = [_mean(runs.get((s, dl), [])) for dl in delays]
        ax.plot(delays, [y if y is not None else float("nan") for y in ys],
                "-o", color=COLORS[s], label=STRAT_LABEL[s])
    ax.set_xlabel("Injected one-way network delay (ms)")
    ax.set_ylabel("Wall-clock to target macro-F1 (s)")
    ax.set_title(f"{ds_name(ds)}: wall-clock to target against slow-tier delay")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(Path(out) / f"fig_delaycurve_{ds}.pdf", bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/mnt/data/ccpe-flids/results/grid")
    ap.add_argument("--out", default="/mnt/data/ccpe-flids/results/figures")
    ap.add_argument("--delaycurve-results",
                     default="/mnt/data/ccpe-flids/results/fix_delaycurve")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results)
    datasets = sorted({r["dataset"] for r in runs})
    print(f"loaded {len(runs)} runs, datasets={datasets}")
    for ds in datasets:
        # NOTE: only the time-to-target-derived figures are regenerated here
        # (crossover, noniid, delaycurve). fig_comm and fig_curves are not
        # t2t-derived (best-F1/uplink and raw convergence curves respectively)
        # and are intentionally left untouched.
        fig_crossover(runs, ds, args.out)
        fig_noniid(runs, ds, args.out)
        print(f"  figures for {ds}")
    if Path(args.delaycurve_results).exists():
        fig_delaycurve(args.delaycurve_results, args.out)
        print("  fig_delaycurve_cicids2017")
    print(f"wrote figures -> {args.out}")


if __name__ == "__main__":
    main()
