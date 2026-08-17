#!/usr/bin/env bash
# Re-run every FedBuff comparison against FedBuff AS PUBLISHED.
#
# Our original "fedbuff" baseline omitted the staleness scaling that Nguyen
# et al. (2022, Sec. 5) specify -- s(tau) = (1+tau)^-0.5, the same discount
# FedAsync uses. That made the published method look like a plain buffered
# average and made our own rule's delta look larger than it is. The corrected
# `fedbuff` strategy now applies that discount; `fedbuff_ns` keeps the old
# staleness-agnostic behaviour as an explicit ablation.
#
# Every claim in the paper that compares our rule to FedBuff is re-measured
# here: the ceiling/reachability sweep, the eta sweep, the K sweep, and the
# main crossover table.
set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/fedbuff_fix.log
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

# 1) ceiling / reachability sweep  (Sec. 6.3): unreachable target, full curve
echo "=== sweep: ceiling + reachability ===" | tee -a "$LOG"
for A in a0.1 a0.5; do
  for S in fedbuff fedbuff_ns; do
    for SEED in 0 1 2; do
      run "fbfix_sweep/cicids2017__${S}__${A}__s${SEED}" \
        --dataset cicids2017 --part-tag "${A}_${SEED}" --strategy "$S" \
        --hardware hetero --network mixed --compression none \
        --target-f1 0.99 --max-seconds 450 --max-version 600
    done
  done
done

# 2) main crossover table (Table 1): LAN and mixed at alpha=0.5
echo "=== crossover table ===" | tee -a "$LOG"
for NET in lan mixed; do
  for SEED in 0 1 2; do
    run "fbfix_cross/cicids2017__fedbuff__${NET}__s${SEED}" \
      --dataset cicids2017 --part-tag "a0.5_${SEED}" --strategy fedbuff \
      --hardware hetero --network "$NET" --compression none --target-f1 0.60
  done
done

# 3) server learning-rate sweep (Sec. 6.4)
echo "=== eta sweep ===" | tee -a "$LOG"
for ETA in 0.1 0.3 0.5 1.0; do
  for SEED in 0 1 2; do
    run "fbfix_eta/cicids2017__fedbuff__eta${ETA}__s${SEED}" \
      --dataset cicids2017 --part-tag "a0.5_${SEED}" --strategy fedbuff \
      --hardware hetero --network mixed --compression none \
      --target-f1 0.60 --server-lr-override "$ETA" --max-seconds 600
  done
done

# 4) buffer-size sweep (Sec. 6.4)
echo "=== K sweep ===" | tee -a "$LOG"
for K in 2 4 8 16; do
  for SEED in 0 1 2; do
    run "fbfix_bufk/cicids2017__fedbuff__K${K}__s${SEED}" \
      --dataset cicids2017 --part-tag "a0.5_${SEED}" --strategy fedbuff \
      --hardware hetero --network mixed --compression none \
      --target-f1 0.60 --buffer-k "$K" --max-seconds 600
  done
done

touch "$R/fedbuff_fix.DONE"
echo "=== ALL DONE ($(date)) ===" | tee -a "$LOG"
