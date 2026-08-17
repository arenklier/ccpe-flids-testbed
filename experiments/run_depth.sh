#!/usr/bin/env bash
# Multi-dataset depth: replicate the two strongest strengthening experiments on
# N-BaIoT and Bot-IoT so the findings are not CICIDS2017-only. Run SEQUENTIALLY.
set -uo pipefail
cd /mnt/data/ccpe-flids
R="python3 -m experiments.run_experiment"

# per-dataset reachable target (small AND large reach it, so times compare)
declare -A TGT=( [nbaiot]=0.60 [botiot]=0.55 )

echo "===== BATCH 1: heavy-model crossover induction (nbaiot, botiot) ====="
# Prediction: on these near-separable, few-round datasets the small model shows
# no crossover (sync fastest on mixed), but the large model raises the slow
# tier's per-round burden and should induce the crossover (staleness overtakes
# sync on mixed) -- a within-dataset causal test of the mechanism.
for ds in nbaiot botiot; do
  for size in small large; do
    for net in lan mixed; do
      for strat in sync staleness; do
        for s in 0 1 2; do
          $R --exp-id depth_model/${ds}__${strat}__${size}__${net}__s${s} \
             --dataset $ds --part-tag a0.5_${s} --strategy $strat \
             --hardware hetero --network $net --compression none \
             --model-size $size --target-f1 ${TGT[$ds]} \
             --max-seconds 500 --max-version 500
        done
      done
    done
  done
done

echo "===== BATCH 2: delay-curve on N-BaIoT (extended 1-500ms) ====="
# Prediction: N-BaIoT's small per-round burden pushes its crossover to a much
# higher delay than CICIDS2017's ~150ms (or out of range entirely).
for delay in 1 50 150 300 500; do
  for strat in sync staleness; do
    for s in 0 1 2; do
      $R --exp-id depth_delaycurve/nbaiot__${strat}__d${delay}__s${s} \
         --dataset nbaiot --part-tag a0.5_${s} --strategy $strat \
         --hardware hetero --network $delay --compression none \
         --target-f1 0.60 --max-seconds 500 --max-version 500
    done
  done
done

touch results/depth.DONE
echo "ALL_DEPTH_DONE"
