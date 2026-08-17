#!/usr/bin/env bash
# Close out the three open questions flagged in the paper's Limitations.
#
#   E1  delay-curve infill      : locate the crossover instead of bracketing it
#                                 between 150 and 250 ms (18 runs)
#   E2  tab:depth repeat-variance: settle the small-model N-BaIoT/Bot-IoT
#                                 ordering that three separate batches
#                                 disagreed on (36 runs)
#   E3  FedAsync anomaly traces : instrumented runs (staleness distribution +
#                                 per-push arrival log) for the two behaviours
#                                 we could not explain (12 runs)
#
# Sequential by design: wall-clock is the metric, so nothing may run
# concurrently. Needs SUDO_PASS exported for the netem (mixed / numeric
# network) runs; LAN runs do not use it.
#
# Usage:  export SUDO_PASS='...' && bash experiments/run_closeout.sh
set -uo pipefail

cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/closeout.log
: > "$LOG"

run() {  # run <exp-id> <args...>
  local id="$1"; shift
  if [ -f "$R/$id/metrics.json" ]; then
    echo "[skip] $id (already done)" | tee -a "$LOG"; return 0
  fi
  echo "[run ] $id" | tee -a "$LOG"
  python3 -m experiments.run_experiment --exp-id "$id" "$@" >>"$LOG" 2>&1 \
    || echo "[FAIL] $id" | tee -a "$LOG"
}

echo "=== E1: delay-curve infill (175/200/225 ms) ===" | tee -a "$LOG"
for D in 175 200 225; do
  for S in sync staleness; do
    for SEED in 0 1 2; do
      run "closeout_delay/cicids2017__${S}__d${D}__s${SEED}" \
        --dataset cicids2017 --part-tag "a0.5_${SEED}" --strategy "$S" \
        --hardware hetero --network "$D" --compression none \
        --model-size small --target-f1 0.60 --max-seconds 400
    done
  done
done

echo "=== E2: small-model repeat variance (N-BaIoT / Bot-IoT) ===" | tee -a "$LOG"
for REP in 0 1 2; do
  for SEED in 0 1 2; do
    for S in sync staleness; do
      run "closeout_depth/nbaiot__${S}__s${SEED}__r${REP}" \
        --dataset nbaiot --part-tag "a0.5_${SEED}" --strategy "$S" \
        --hardware hetero --network mixed --compression none \
        --model-size small --target-f1 0.60 --max-seconds 600
      run "closeout_depth/botiot__${S}__s${SEED}__r${REP}" \
        --dataset botiot --part-tag "a0.5_${SEED}" --strategy "$S" \
        --hardware hetero --network mixed --compression none \
        --model-size small --target-f1 0.55 --max-seconds 600
    done
  done
done

echo "=== E3: FedAsync anomaly traces ===" | tee -a "$LOG"
# (a) N-BaIoT LAN vs mixed: LAN was slower than mixed in all three seeds
for NET in lan mixed; do
  for SEED in 0 1 2; do
    run "closeout_anom/nbaiot__fedasync__${NET}__s${SEED}" \
      --dataset nbaiot --part-tag "a0.5_${SEED}" --strategy fedasync \
      --hardware hetero --network "$NET" --compression none \
      --model-size small --target-f1 0.68 --max-seconds 900
  done
done
# (b) CICIDS2017 mixed: FedAsync is slowest of the four at IID, not at skew
for TAG in iid a0.5; do
  for SEED in 0 1 2; do
    run "closeout_anom/cicids2017__fedasync__${TAG}__s${SEED}" \
      --dataset cicids2017 --part-tag "${TAG}_${SEED}" --strategy fedasync \
      --hardware hetero --network mixed --compression none \
      --model-size small --target-f1 0.60 --max-seconds 900
  done
done

touch "$R/closeout.DONE"
echo "=== ALL DONE ($(date)) ===" | tee -a "$LOG"
