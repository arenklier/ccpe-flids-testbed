#!/usr/bin/env bash
# Five follow-up experiment batches addressing editor review, run SEQUENTIALLY
# (shared host, wall-clock is the metric — never parallelize with anything else).
set -uo pipefail
cd /mnt/data/ccpe-flids
R="python3 -m experiments.run_experiment"

echo "===== BATCH 1: round-count causal test (local_steps knob, dataset fixed) ====="
# Fewer local steps per push -> more rounds needed for the same amount of
# training -> more barrier crossings for sync. Holds dataset/data-heterogeneity
# fixed; only varies how many communication rounds the workload needs.
for net in lan mixed; do
  for steps in 10 30 90; do
    for strat in sync fedbuff fedasync staleness; do
      for s in 0 1 2; do
        $R --exp-id fix_roundcount/cicids2017__${strat}__steps${steps}__${net}__s${s} \
           --dataset cicids2017 --part-tag a0.5_${s} --strategy $strat \
           --hardware hetero --network $net --compression none \
           --target-f1 0.60 --max-seconds 500 --max-version 500 \
           --local-steps $steps
      done
    done
  done
done

echo "===== BATCH 2: staleness generalization (nbaiot, botiot, alpha 0.1 + 0.5) ====="
# Same target-threshold-sweep design as the CICIDS2017 flagship result,
# extended to the other two datasets to test generalization.
for ds in nbaiot botiot; do
  for ni in a0.1 a0.5; do
    for strat in sync fedbuff fedasync staleness; do
      for s in 0 1 2; do
        $R --exp-id fix_generalize/${ds}__${strat}__${ni}__s${s} \
           --dataset $ds --part-tag ${ni}_${s} --strategy $strat \
           --hardware hetero --network mixed --compression none \
           --target-f1 0.99 --max-seconds 300 --max-version 400
      done
    done
  done
done

echo "===== BATCH 3: eta (server_lr) sensitivity, fedbuff vs staleness ====="
for eta in 0.1 0.3 0.5 1.0; do
  for strat in fedbuff staleness; do
    for s in 0 1 2; do
      $R --exp-id fix_eta/cicids2017__${strat}__eta${eta}__s${s} \
         --dataset cicids2017 --part-tag a0.5_${s} --strategy $strat \
         --hardware hetero --network mixed --compression none \
         --target-f1 0.60 --max-seconds 400 --max-version 500 \
         --server-lr-override $eta
    done
  done
done

echo "===== BATCH 4: buffer K sensitivity, fedbuff vs staleness ====="
for k in 2 4 8 16; do
  for strat in fedbuff staleness; do
    for s in 0 1 2; do
      $R --exp-id fix_bufk/cicids2017__${strat}__k${k}__s${s} \
         --dataset cicids2017 --part-tag a0.5_${s} --strategy $strat \
         --hardware hetero --network mixed --compression none \
         --target-f1 0.60 --max-seconds 400 --max-version 500 \
         --buffer-k $k
    done
  done
done

echo "===== BATCH 5: continuous network-severity crossover curve (sync vs staleness) ====="
for delay in 1 25 50 100 150 250; do
  for strat in sync staleness; do
    for s in 0 1 2; do
      $R --exp-id fix_delaycurve/cicids2017__${strat}__d${delay}__s${s} \
         --dataset cicids2017 --part-tag a0.5_${s} --strategy $strat \
         --hardware hetero --network $delay --compression none \
         --target-f1 0.60 --max-seconds 400 --max-version 500
    done
  done
done

touch results/fixes.DONE
echo "ALL_FIXES_DONE"
