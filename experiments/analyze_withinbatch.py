"""Produce every number the paper needs from the within-batch re-measurement.

All four strategy comparisons in the paper were originally measured against a
FedBuff baseline that omitted the staleness discount the published method
specifies. This script recomputes them from runs that all come from a single
batch, so nothing is compared across batches.

Time-to-target is linearly interpolated between the evaluation checkpoints
that bracket the target, which corrects for the different checkpoint spacing
across strategies (see the paper's Limitations).

Usage (on the server):
  python3 experiments/analyze_withinbatch.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
from collections import defaultdict

R = "/mnt/data/ccpe-flids/results"
ARMS = ["sync", "fedasync", "fedbuff", "fedbuff_ns", "staleness"]
LABEL = {"sync": "Sync FedAvg", "fedasync": "FedAsync", "fedbuff": "FedBuff",
         "fedbuff_ns": "Buffered-NS (abl.)", "staleness": "Ours (+data size)"}


def interp_t2t(hist: list[dict], target: float) -> float | None:
    """Wall-clock at which macro-F1 first crosses `target`, interpolated."""
    if not any(h["macro_f1"] >= target for h in hist):
        return None
    for i in range(1, len(hist)):
        if hist[i]["macro_f1"] >= target and hist[i - 1]["macro_f1"] < target:
            f0, f1 = hist[i - 1]["macro_f1"], hist[i]["macro_f1"]
            t0, t1 = hist[i - 1]["wall_s"], hist[i]["wall_s"]
            frac = (target - f0) / (f1 - f0) if f1 != f0 else 0.0
            return t0 + frac * (t1 - t0)
    return next(h["wall_s"] for h in hist if h["macro_f1"] >= target)


def load(pattern: str, keyfn):
    """Group runs under `pattern` by keyfn(run_dir_name) -> key."""
    out = defaultdict(list)
    for path in glob.glob(f"{R}/{pattern}/*/metrics.json"):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        name = path.split("/")[-2]
        out[keyfn(name)].append(d)
    return out


def fmt(vals, nd=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a".rjust(11)
    return f"{st.mean(vals):.{nd}f}±{st.pstdev(vals):.{nd}f}".rjust(11)


def t2t_of(runs):
    return [interp_t2t(d["history"], d["config"]["target_macro_f1"]) for d in runs]


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------- Table 1 ---
section("TABLE 1  crossover: wall-clock to target (s), CICIDS2017, a=0.5")
cross = load("wb_cross", lambda n: (n.split("__")[1], n.split("__")[2]))
cross.update(load("fbfix_cross", lambda n: ("fedbuff", n.split("__")[2])))
print(f"{'strategy':20s} {'LAN':>11s} {'mixed':>11s}   reached")
for a in ARMS:
    lan, mix = t2t_of(cross.get((a, "lan"), [])), t2t_of(cross.get((a, "mixed"), []))
    nl = sum(x is not None for x in lan)
    nm = sum(x is not None for x in mix)
    print(f"{LABEL[a]:20s} {fmt(lan)} {fmt(mix)}   {nl}/3, {nm}/3")

# --------------------------------------------------------------- control ---
section("BATCH-EFFECT CONTROL  (fedbuff_ns == the old, mis-implemented arm)")
print("If these land near the old grid values (LAN 75, mixed 98), timing is")
print("comparable across batches; if not, only within-batch numbers are valid.")
ns_lan, ns_mix = t2t_of(cross.get(("fedbuff_ns", "lan"), [])), t2t_of(cross.get(("fedbuff_ns", "mixed"), []))
print(f"  fedbuff_ns  LAN {fmt(ns_lan)}  (old grid: 75)")
print(f"  fedbuff_ns  mix {fmt(ns_mix)}  (old grid: 98)")

# ---------------------------------------------------------------- Table 2 ---
section("TABLE 2  ceiling + reachability, CICIDS2017 mixed (unreachable target)")
sweep = load("wb_sweep", lambda n: (n.split("__")[1], n.split("__")[2]))
sweep.update(load("fbfix_sweep", lambda n: (n.split("__")[1], n.split("__")[2])))
for alpha in ("a0.1", "a0.5"):
    print(f"\n  alpha = {alpha[1:]}")
    print(f"  {'strategy':20s} {'ceiling':>13s} {'best':>7s}  >=.64 >=.66 >=.68")
    for a in ARMS:
        runs = sweep.get((a, alpha), [])
        if not runs:
            print(f"  {LABEL[a]:20s} {'n/a':>13s}")
            continue
        bests = [max(h["macro_f1"] for h in d["history"]) for d in runs]
        reach = {t: sum(1 for b in bests if b >= t) for t in (0.64, 0.66, 0.68)}
        print(f"  {LABEL[a]:20s} {fmt(bests, 3)} {max(bests):7.3f}"
              f"   {reach[0.64]}/{len(bests)}  {reach[0.66]}/{len(bests)}  {reach[0.68]}/{len(bests)}")

# ------------------------------------------------------------ sensitivity ---
for tag, pat_new, pat_old, knob in (
        ("ETA sweep (server learning rate)", "wb_eta", "fbfix_eta", "eta"),
        ("K sweep (buffer size)", "wb_bufk", "fbfix_bufk", "K")):
    section(tag)
    d = load(pat_new, lambda n: (n.split("__")[1], n.split("__")[2]))
    d.update(load(pat_old, lambda n: (n.split("__")[1], n.split("__")[2])))
    knobs = sorted({k[1] for k in d}, key=lambda s: float(s.replace(knob, "")))
    print(f"  {knob:>6s}  {'FedBuff':>11s}  {'Ours':>11s}   winner")
    for kv in knobs:
        fb, ours = t2t_of(d.get(("fedbuff", kv), [])), t2t_of(d.get(("staleness", kv), []))
        fbv = [x for x in fb if x is not None]
        ov = [x for x in ours if x is not None]
        win = "-" if not (fbv and ov) else ("FedBuff" if st.mean(fbv) < st.mean(ov) else "Ours")
        print(f"  {kv.replace(knob, ''):>6s}  {fmt(fb)}  {fmt(ours)}   {win}"
              f"   (n={len(fbv)}/3, {len(ov)}/3)")

print()
