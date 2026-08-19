# FL-IDS wall-clock testbed

A container-based testbed for measuring **wall-clock** behaviour of synchronous
and asynchronous federated-learning aggregation on real processes, real
sockets and shaped networks, together with every measurement backing the
paper *Measuring on the Critical Path Inverts the Answer: A Real-Testbed Study
of Synchronous and Asynchronous Federated Learning for Intrusion Detection*.

The short version of what we found: our first, entirely reasonable-looking
harness evaluated the global model **on the aggregator's critical path**, which
charges each strategy in proportion to how often it advances the model. That is
far more often for asynchronous rules than for a synchronous barrier, so the
harness taxed them ~14x against ~1.6x, and the resulting measurements said
synchronous FedAvg was the faster choice on a fast LAN. Moving evaluation onto
a worker thread, changing nothing else, reverses that conclusion. Both the
flawed and the corrected runs are in this repository so the effect can be
checked rather than taken on trust (see [results/MANIFEST.md](results/MANIFEST.md)).

## What is here

```
src/flids/            aggregation server, client loop, compression, data pipeline
experiments/          run scripts (one per batch in the paper) and analysis
testbed/scripts/      tc/netem shaping applied per container tier
results/runs/         raw metrics.json + config.yaml for all 1692 runs
results/figures/      figures as they appear in the paper
results/tables/       LaTeX tables as they appear in the paper
```

## Requirements

A Linux host with Docker, and enough cores that 12 cgroup-limited containers
are not fighting each other (we used a 96-core machine). Containers need
`NET_ADMIN` for `tc`; no root or `sudo` on the host is required, because
shaping is applied with `docker exec` inside each container.

```bash
docker build -t ccpe-flids:latest .
```

Datasets are **not** redistributed here. Obtain CICIDS2017 (we use the
DistriNet cleaned release with the corrected labels of Engelen et al.),
Bot-IoT (5% subset) and N-BaIoT from their original sources, then build the
partitions:

```bash
python3 -m experiments.prepare_data --dataset cicids2017 --raw <path> \
  --out datasets/prepared/cicids2017 --n-clients 12 --alpha 0.5 --seed 0
```

The preprocessing drops identifiers (IPs, ports, timestamps), carves the test
split **before** partitioning so no test row can reach a client shard, and
standardises using training statistics only.

## Running an experiment

```bash
python3 -m experiments.run_experiment \
  --exp-id demo --dataset cicids2017 --part-tag a0.5_0 \
  --strategy staleness --hardware hetero --network mixed \
  --compression none --target-f1 0.60
```

`--strategy` is one of `sync`, `fedasync`, `fedbuff`, `fedbuff_ns`,
`staleness`. `--network` takes `lan`, `mixed`, or an integer, which is read as
the slow tier's one-way delay in milliseconds for a continuous severity sweep.
`--docker-clients` runs only part of the fleet locally so the rest can attach
from another machine; see `experiments/run_crosshost.sh` and
`experiments/real_clients.sh` for the two-machine setup.

## If you are measuring wall-clock yourself

Three properties of a harness changed our conclusions during this work, and
none of them is exotic. They are worth checking in any FL timing study:

1. **Keep evaluation off the critical path.** If the aggregator scores the
   model while holding the lock that serialises client traffic, strategies that
   advance the model more often pay more for it. In our case that was a 14x
   tax on a buffered asynchronous rule against 1.6x on the barrier, which
   inverted the headline. `src/flids/server/app.py` now snapshots parameters
   under the lock and scores them on a worker thread.

2. **Space checkpoints by time, not by model version.** A fixed
   `eval_every` in versions spaces checkpoints 3-4x further apart in wall-clock
   for a barrier than for an asynchronous rule, which biases time-to-target by
   2-24%. We interpolate the crossing between the checkpoints that bracket the
   target; `experiments/analyze_withinbatch.py` shows the calculation.

3. **Run the arms you are comparing adjacently.** Re-running one unchanged
   configuration an hour later gave 0.91 instead of 2.04 seconds per model
   version, with the same number of versions to target: the host had changed,
   not the workload. A batch that loops over strategies outermost can separate
   the arms of a comparison by hours, which is enough for that drift to
   dominate the difference being reported. Every script here puts strategy in
   the innermost loop.

We would add that run order deserves the same place in a reproducibility
statement that random seeds already occupy.

## Archived version

This repository is the development copy. The archived snapshot that backs the
paper, with the raw output of all 1692 runs, is at
[doi:10.5281/zenodo.22012392](https://doi.org/10.5281/zenodo.22012392).

## Licence

MIT, see [LICENSE](LICENSE). The datasets are covered by their own terms.
