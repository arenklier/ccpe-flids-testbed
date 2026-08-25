"""Launch one testbed run: aggregator + N heterogeneous client containers.

Each client gets its own cgroup CPU/memory limit (hardware tier) and its own
rank/id/data shard. Network shaping (tc netem) is applied only for the 'mixed'
profile via testbed/scripts/apply_netem.sh (needs root). The aggregator exits
when it hits the target macro-F1 or the wall-clock budget; we then collect its
metrics.json.

Example:
  python -m experiments.run_experiment \
    --exp-id demo --dataset nbaiot --part-tag a0.5_0 \
    --strategy staleness --hardware hetero --network lan \
    --compression none
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

DATA_ROOT = "/mnt/data/ccpe-flids/datasets"
RESULTS_ROOT = "/mnt/data/ccpe-flids/results"
NET = "flnet"
IMAGE = "ccpe-flids:latest"

TIERS = {  # cpus, memory
    "fast": ("4.0", "8g"),
    "mid":  ("2.0", "4g"),
    "slow": ("0.5", "2g"),
}


def sh(*a, **k):
    return subprocess.run(a, check=True, text=True, capture_output=True, **k)


def tier_for(rank: int, hardware: str, tier_mix: str | None = None) -> str:
    if hardware == "homo":
        return "fast"
    if tier_mix:
        nf, nm, ns = (int(x) for x in tier_mix.split(","))
        if rank < nf:
            return "fast"
        if rank < nf + nm:
            return "mid"
        return "slow"
    return ["fast", "mid", "slow"][rank % 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--part-tag", required=True, help="e.g. a0.5_0 or iid_0")
    ap.add_argument("--strategy", required=True,
                    choices=["sync", "fedasync", "fedbuff", "fedbuff_ns", "staleness"])
    ap.add_argument("--hardware", default="hetero", choices=["homo", "hetero"])
    ap.add_argument("--network", default="lan",
                    help="'lan', 'mixed' (fixed 50/150ms), or an integer string "
                         "= slow-tier delay in ms for a continuous severity sweep "
                         "(e.g. '25', '100') — see testbed/scripts/apply_netem.sh")
    ap.add_argument("--compression", default="none",
                    choices=["none", "int8", "topk"])
    ap.add_argument("--n-clients", type=int, default=12)
    ap.add_argument("--target-f1", type=float, default=0.80)
    ap.add_argument("--max-version", type=int, default=2000)
    ap.add_argument("--max-seconds", type=int, default=3600)
    ap.add_argument("--local-steps", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-cap", type=int, default=0,
                    help="test rows per class for cadence checkpoints; 0 = full split")
    ap.add_argument("--server-lr-override", type=float, default=None,
                    help="override the strategy-default server_lr (sensitivity sweep)")
    ap.add_argument("--buffer-k", type=int, default=4,
                    help="buffer size for fedbuff/staleness (sensitivity sweep)")
    ap.add_argument("--model-size", default="small", choices=["small", "large"],
                    help="MLP hidden sizes; 'large' stress-tests the slow tier's compute burden")
    ap.add_argument("--docker-clients", type=int, default=None,
                    help="how many of --n-clients run as local containers; the "
                         "rest are expected to attach from another machine "
                         "(cross-host validation). Defaults to all of them.")
    ap.add_argument("--publish-port", action="store_true",
                    help="publish the aggregator's port on the host so clients "
                         "on other machines can reach it")
    ap.add_argument("--tier-mix", default=None,
                    help="comma-separated fast,mid,slow counts overriding the default "
                         "even 3-way split (e.g. '2,2,8' for a slow-heavy fleet); "
                         "counts must sum to --n-clients")
    args = ap.parse_args()

    part = f"{DATA_ROOT}/prepared/{args.dataset}/{args.part_tag}"
    meta_path = Path(part) / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"missing meta.json under {part} — run prepare_data first")
    meta = json.loads(meta_path.read_text())

    run_name = f"{args.exp_id}"
    out_host = f"{RESULTS_ROOT}/{run_name}"          # path may contain '/'
    safe = run_name.replace("/", "-")                # docker names forbid '/'
    Path(out_host).mkdir(parents=True, exist_ok=True)

    # server_lr is the effective mixing weight. Only synchronous FedAvg averages
    # full-participation deltas, where eta=1 is the standard step. ALL async
    # variants (FedAsync single-update, FedBuff and our staleness-aware buffered
    # update) share eta=0.3 — this keeps FedBuff vs staleness a clean, single-
    # variable comparison so any gap is attributable to the staleness weighting,
    # not the learning rate.
    server_lr = 1.0 if args.strategy == "sync" else 0.3
    if args.server_lr_override is not None:
        server_lr = args.server_lr_override

    cfg = {
        "strategy": args.strategy, "n_clients": args.n_clients,
        "in_dim": meta["in_dim"], "n_classes": meta["n_classes"],
        "test_path": f"/part/test.parquet", "out_dir": "/out",
        "compression": args.compression, "target_macro_f1": args.target_f1,
        "max_version": args.max_version, "max_seconds": args.max_seconds,
        "eval_every": args.eval_every, "eval_cap": args.eval_cap, "buffer_k": args.buffer_k, "staleness_a": 0.5,
        "server_lr": server_lr, "model_size": args.model_size,
        "dataset": args.dataset, "part_tag": args.part_tag,
        "hardware": args.hardware, "network": args.network,
    }
    Path(out_host, "config.yaml").write_text(yaml.safe_dump(cfg))

    # fresh network + clean any stale containers
    subprocess.run(["docker", "network", "create", NET], capture_output=True)
    n_local_names = args.docker_clients if args.docker_clients is not None else args.n_clients
    names = [f"{safe}-agg"] + [f"{safe}-c{r:02d}" for r in range(n_local_names)]
    subprocess.run(["docker", "rm", "-f", *names], capture_output=True)

    # aggregator
    agg_cmd = ["docker", "run", "-d", "--name", names[0], "--network", NET,
               "--network-alias", "aggregator"]
    if args.publish_port:
        agg_cmd += ["-p", "8080:8080"]
    sh(*agg_cmd,
       "-v", f"{part}:/part:ro", "-v", f"{out_host}:/out",
       "-v", f"{out_host}:/cfgdir",
       IMAGE, "python", "-m", "flids.server.main", "--config", "/cfgdir/config.yaml")
    time.sleep(4)  # let uvicorn bind

    # clients: only the local share when the rest attach from elsewhere
    n_local = args.n_clients if args.docker_clients is None else args.docker_clients
    for r in range(n_local):
        tier = tier_for(r, args.hardware, args.tier_mix)
        cpus, mem = TIERS[tier]
        sh("docker", "run", "-d", "--name", names[r + 1], "--network", NET,
           "--cpus", cpus, "--memory", mem, "--cap-add", "NET_ADMIN",
           "-v", f"{part}:/part:ro",
           "-e", "SERVER_ADDR=aggregator:8080",
           "-e", f"CLIENT_ID=c{r:02d}", "-e", f"CLIENT_RANK={r}",
           "-e", f"TIER={tier}", "-e", f"CPU_BUDGET={cpus}",
           "-e", f"OMP_NUM_THREADS={max(1, int(float(cpus)))}",
           "-e", "PARTITION_DIR=/part",
           "-e", f"IN_DIM={meta['in_dim']}", "-e", f"N_CLASSES={meta['n_classes']}",
           "-e", f"LOCAL_STEPS={args.local_steps}", "-e", "LOCAL_LR=0.05",
           "-e", f"COMPRESSION={args.compression}", "-e", f"MODEL_SIZE={args.model_size}",
           IMAGE, "python", "-m", "flids.client.main", "--tier", tier)

    if args.network != "lan":
        # Shaping happens inside the client containers, which are started with
        # --cap-add NET_ADMIN and ship iproute2, so no host root (and no
        # password) is involved.
        script = str(Path(__file__).parents[1] / "testbed/scripts/apply_netem.sh")
        r = subprocess.run(["bash", script, args.network, safe],
                           text=True, capture_output=True)
        print(r.stdout, end="")
        if r.returncode != 0:
            print(f"[netem] WARNING apply failed: {r.stderr.strip()}")

    print(f"[run {run_name}] {args.strategy}/{args.hardware}/{args.network}/"
          f"{args.compression} launched; waiting for aggregator...")

    # wait for aggregator to exit
    while True:
        st = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}",
                             names[0]], capture_output=True, text=True)
        if st.stdout.strip() != "true":
            break
        time.sleep(3)

    subprocess.run(["docker", "logs", "--tail", "20", names[0]])
    subprocess.run(["docker", "rm", "-f", *names], capture_output=True)
    print(f"[run {run_name}] done -> {out_host}/metrics.json")


if __name__ == "__main__":
    main()
