# Which runs back which claim

Every run directory under `runs/` holds a `config.yaml` (the exact settings)
and a `metrics.json` (evaluation history, final version count, and, for later
batches, a per-push trace with staleness). Wall-clock in the paper is the
linearly interpolated crossing of the target macro-F1 between the two
evaluation checkpoints that bracket it; `../experiments/analyze_withinbatch.py`
implements it.

## Runs behind the paper's tables and figures

These were all produced **after** evaluation was moved off the aggregator's
critical path, and each batch runs the strategies it compares back to back.

| Directory | Backs |
|---|---|
| `fin_cross/` | Table 1, crossover across three datasets, LAN vs mixed |
| `finsweep/` | Table 2, quality reachable within an equal 450 s budget |
| `t3/` | Table 3, communication vs accuracy under int8 and top-k |
| `fin_depth/` | Table 4, small vs large model |
| `finscale/` | Table 5, 12 to 96 clients |
| `fin_delay/` | delay-curve figure, nine slow-tier delays |
| `finsens_eta/`, `finsens_k/` | server learning rate and buffer size sweeps |
| `xh/` | two-machine validation, 8 containerised + 4 real clients |
| `smoke_offlock/` | the corrected side of the 14x evaluation-tax measurement |

## Runs kept as evidence, not as results

The paper's central claim is that a harness detail reversed our conclusions, so
the superseded measurements are part of the evidence rather than clutter. They
were produced with evaluation **on** the critical path and their wall-clock
numbers should not be compared against the batches above.

| Directory | Why it is here |
|---|---|
| `grid/` | the original 576-cell grid; strategy was the outer loop, so its arms are also separated by hours and confounded with host drift |
| `t1cic/` | same configuration as `smoke_offlock`, measured inline: the two together give the 14x versus 1.6x tax |
| `fbfix_*/` | first correction of the FedBuff baseline, which had omitted the staleness discount the published method specifies |
| `wb_*/`, `t1rest/`, `ctl_*/` | within-batch re-measurements used to establish the 2.2x host drift and to separate it from the FedBuff fix |
| `sweep/`, `fix_*/`, `depth_*/`, `strengthen_*/`, `closeout_*/` | earlier sweeps whose qualitative findings survived but whose absolute timings did not |
| `crosshost_*/`, `real_crosshost/` | the two-machine trials run before the fix, which reproduced the harness bug faithfully and reported the inverted ordering |

## Baselines

`baselines.json` holds, per dataset, the majority-class macro-F1 floor and a
centralised upper bound. Every federated run in the paper sits between them.
