#!/usr/bin/env bash
# Within-batch re-measurement of every strategy comparison.
#
# The FedBuff correction (adding the staleness discount the published method
# specifies) was measured in a separate batch from the numbers already in the
# paper, and our own fedbuff_ns control showed a batch-to-batch shift large
# enough that cross-batch comparison is not safe. This script re-runs the
# missing arms so that every strategy in every table comes from one batch,
# under identical conditions.
#
# Arms: sync, fedasync, fedbuff (as published), fedbuff_ns (ablation:
# data-size weights, no staleness), staleness (ours).
# fedbuff arms already exist from run_fedbuff_fix.sh and are skipped.
set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/withinbatch.log
: > "$LOG"

run() {
  local id="$1"; shift
  if [ -f "$R/$id/metrics.json" ]; then
    echo "[skip] $id" | tee -a "$LOG"; return 0
  fi
  echo "[run ] $id" | tee -a "$LOG"
  python3 -m experiments.run_experiment --exp-id "$id" "$@" >>"$LOG" 2>&1 \
    || echo "[FAIL] $id" | tee -a "$LOG"
}

# A. crossover table: LAN + mixed, alpha=0.5, reachable target
echo "=== A: crossover table ===" | tee -a "$LOG"
for NET in lan mixed; do
  for S in sync fedasync fedbuff_ns staleness; do
    for SEED in 0 1 2; do
      run "wb_cross/cicids2017__${S}__${NET}__s${SEED}" \
        --dataset cicids2017 --part-tag "a0.5_${SEED}" --strategy "$S" \
        --hardware hetero --network "$NET" --compression none --target-f1 0.60
    done
  done
done

# C. eta sweep (cheap) -- ours, to pair with the corrected FedBuff arm
echo "=== C: eta sweep ===" | tee -a "$LOG"
for ETA in 0.1 0.3 0.5 1.0; do
  for SEED in 0 1 2; do
    run "wb_eta/cicids2017__staleness__eta${ETA}__s${SEED}" \
      --dataset cicids2017 --part-tag "a0.5_${SEED}" --strategy staleness \
      --hardware hetero --network mixed --compression none \
      --target-f1 0.60 --server-lr-override "$ETA" --max-seconds 600
  done
done

# D. buffer-size sweep
echo "=== D: K sweep ===" | tee -a "$LOG"
for K in 2 4 8 16; do
  for SEED in 0 1 2; do
    run "wb_bufk/cicids2017__staleness__K${K}__s${SEED}" \
      --dataset cicids2017 --part-tag "a0.5_${SEED}" --strategy staleness \
      --hardware hetero --network mixed --compression none \
      --target-f1 0.60 --buffer-k "$K" --max-seconds 600
  done
done

# B. ceiling / reachability sweep (slowest: unreachable target, full budget)
echo "=== B: ceiling + reachability sweep ===" | tee -a "$LOG"
for A in a0.1 a0.5; do
  for S in sync fedasync staleness; do
    for SEED in 0 1 2; do
      run "wb_sweep/cicids2017__${S}__${A}__s${SEED}" \
        --dataset cicids2017 --part-tag "${A}_${SEED}" --strategy "$S" \
        --hardware hetero --network mixed --compression none \
        --target-f1 0.99 --max-seconds 450 --max-version 600
    done
  done
done

touch "$R/withinbatch.DONE"
echo "=== ALL DONE ($(date)) ===" | tee -a "$LOG"
