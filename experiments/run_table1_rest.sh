#!/usr/bin/env bash
# Complete Table 1 (N-BaIoT, Bot-IoT) with a drift-safe run order.
#
# Timestamps on the original grid show strategy was the OUTER loop, so the
# four strategies for one condition were measured up to four hours apart --
# and a same-configuration control showed host throughput drifting by 2.2x
# over that timescale (2.04 vs 0.91 s per server version). Any cross-strategy
# wall-clock comparison built that way is confounded with drift.
#
# Here strategy is the INNERMOST loop: the four strategies being compared run
# back to back within roughly two minutes of each other, so drift cannot
# separate them. Seeds and networks vary outside that.
set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/table1rest.log
: > "$LOG"

target_for() { case "$1" in nbaiot) echo 0.68 ;; botiot) echo 0.62 ;; esac; }

for DS in nbaiot botiot; do
  T=$(target_for "$DS")
  for NET in lan mixed; do
    for SEED in 0 1 2; do
      for S in sync fedasync fedbuff staleness; do
        ID="t1rest/${DS}__${S}__${NET}__s${SEED}"
        if [ -f "$R/$ID/metrics.json" ]; then
          echo "[skip] $ID" | tee -a "$LOG"; continue
        fi
        echo "[run ] $ID" | tee -a "$LOG"
        python3 -m experiments.run_experiment --exp-id "$ID" \
          --dataset "$DS" --part-tag "a0.5_${SEED}" --strategy "$S" \
          --hardware hetero --network "$NET" --compression none \
          --target-f1 "$T" >>"$LOG" 2>&1 || echo "[FAIL] $ID" | tee -a "$LOG"
      done
    done
  done
done

touch "$R/table1rest.DONE"
echo "=== ALL DONE ($(date)) ===" | tee -a "$LOG"
