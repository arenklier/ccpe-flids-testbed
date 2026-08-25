# Which runs back which claim

Every run directory under `runs/` holds a `config.yaml` (the exact settings)
and a `metrics.json` (evaluation history, final version count, and, for later
batches, a per-push trace with staleness). Wall-clock in the paper is the
linearly interpolated crossing of the target macro-F1 between the two
evaluation checkpoints that bracket it; `../experiments/analyze_withinbatch.py`
implements it.

## Runs behind the paper's tables and figures

These were produced with evaluation off the aggregator's critical path **and**
with the class-stratified checkpoint set, so one evaluator is used throughout.
Each batch runs the strategies it compares back to back.

| Directory | Backs |
|---|---|
| `fin_cross2/` | Table 1 and Figures 1 and 4, crossover across three datasets, LAN vs mixed |
| `finsweep_f/` | Table 2, quality reachable within an equal 450 s budget |
| `t3_f/` | Table 3 and Figure 5, communication vs accuracy under int8 and top-k |
| `fin_steps_f/` | Table 4, local work per push from 10 to 120 minibatch steps |
| `fin_depth_f/` | Table 5, small vs large model |
| `finscale_f/` | Table 6, 12 to 96 clients |
| `fin_skew_f/` | Table 7, shard-size skew manipulated directly at four ratios |
| `fin_beta/`, `fin_clip/` | Table 8, sweeping the strength of the data-size term and a median-clipped variant |
| `fin_lock/` | aggregator lock wait and hold time, which bounds how much of the barrier's cost could be contention |
| `fin_delay_f/` | Figure 2, nine slow-tier delays |
| `fin_noniid_f/` | Figure 3, label-skew sweep |
| `fin_tier_f/` | tier-ratio robustness, 2:2:8 and 8:2:2 splits |
| `fin_depthlan_f/` | small vs large model on a LAN, isolating pure compute cost |
| `fin_ns_f/` | buffered-NS ablation, which term of the buffer weight does the damage |
| `finsens_eta_f/`, `finsens_k_f/` | server learning rate and buffer size, CICIDS2017 |
| `fin_delay2_f/`, `finsens_eta2_f/`, `finsens_k2_f/` | the same sweeps on N-BaIoT and Bot-IoT |
| `xh/` | two-machine validation, 8 containerised + 4 real clients. This one needs a
second physical machine and could not be re-measured with the stratified
evaluator; both of its arms use the coarse one, which the paper states. |
| `smoke_offlock/` | the corrected side of the 14x evaluation-tax measurement |

The `_f` suffix marks the stratified-checkpoint re-measurement. The
corresponding directories without it are the earlier, coarser measurements of
the same configurations, kept for the comparison in Section 6.

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
| `fin_cross/`, `finsweep/`, `t3/`, `fin_steps/`, `fin_depth/`, `finscale/`, `fin_skew/`, `fin_delay/`, `fin_noniid/`, `fin_tier/`, `fin_depthlan/`, `fin_ns/`, `finsens_*/`, `fin_delay2/` | the same configurations measured before the stratified checkpoint set, when a fast run could record as few as two checkpoints. Their asynchronous times are biased late; the paper quantifies the shift |

## Baselines

`baselines.json` holds, per dataset, the majority-class macro-F1 floor and a
centralised upper bound. Every federated run in the paper sits between them.
