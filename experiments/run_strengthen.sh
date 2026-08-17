#!/usr/bin/env bash
# Three follow-up robustness experiments for CCPE-specific critique points,
# run SEQUENTIALLY (wall-clock is the metric).
set -uo pipefail
cd /mnt/data/ccpe-flids
R="python3 -m experiments.run_experiment"

echo "===== BATCH A: heavier model (does the crossover survive a bigger classifier?) ====="
for net in lan mixed; do
  for strat in sync staleness; do
    for s in 0 1 2; do
      $R --exp-id strengthen_model/cicids2017__${strat}__${net}__s${s} \
         --dataset cicids2017 --part-tag a0.5_${s} --strategy $strat \
         --hardware hetero --network $net --compression none \
         --target-f1 0.50 --max-seconds 600 --max-version 300 \
         --model-size large
    done
  done
done

echo "===== BATCH B: hardware tier-ratio sensitivity (is 4:4:4 cherry-picked?) ====="
for mix in "2,2,8" "8,2,2"; do
  for net in lan mixed; do
    for strat in sync staleness; do
      for s in 0 1 2; do
        tag=$(echo $mix | tr ',' '-')
        $R --exp-id strengthen_tiermix/cicids2017__${strat}__${net}__mix${tag}__s${s} \
           --dataset cicids2017 --part-tag a0.5_${s} --strategy $strat \
           --hardware hetero --network $net --compression none \
           --target-f1 0.60 --max-seconds 400 --max-version 400 \
           --tier-mix $mix
      done
    done
  done
done

echo "===== BATCH C: parameter-server scalability (12/24/48/96 clients, sync, LAN, homo) ====="
$R --exp-id strengthen_scale/nc12 --dataset cicids2017 --part-tag a0.5_0 \
   --strategy sync --hardware homo --network lan --compression none \
   --target-f1 0.60 --max-seconds 300 --max-version 200 --n-clients 12
$R --exp-id strengthen_scale/nc24 --dataset cicids2017_nc24 --part-tag a0.5_0 \
   --strategy sync --hardware homo --network lan --compression none \
   --target-f1 0.60 --max-seconds 300 --max-version 200 --n-clients 24
$R --exp-id strengthen_scale/nc48 --dataset cicids2017_nc48 --part-tag a0.5_0 \
   --strategy sync --hardware homo --network lan --compression none \
   --target-f1 0.60 --max-seconds 300 --max-version 200 --n-clients 48
$R --exp-id strengthen_scale/nc96 --dataset cicids2017_nc96 --part-tag a0.5_0 \
   --strategy sync --hardware homo --network lan --compression none \
   --target-f1 0.60 --max-seconds 300 --max-version 200 --n-clients 96

touch results/strengthen.DONE
echo "ALL_STRENGTHEN_DONE"
