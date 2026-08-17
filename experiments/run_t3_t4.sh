#!/usr/bin/env bash
# Re-measure Table 3 (communication) and Table 4 (heavy-model induction).
#
# Two independent reasons to redo these:
#   * their FedBuff rows were produced by the mis-implemented baseline that
#     omitted the published staleness discount;
#   * the original grid ran strategy as the outer loop, so strategies in one
#     condition were measured hours apart, and a same-configuration control
#     showed host throughput drifting 2.2x over that timescale.
#
# Strategy is therefore the INNERMOST loop here: every set of strategies being
# compared runs back to back. Waits for the Table 1 completion job first --
# wall-clock is the metric, so nothing may run concurrently.
set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/t3t4.log
: > "$LOG"

echo "waiting for table1rest to finish..." | tee -a "$LOG"
while [ ! -f "$R/table1rest.DONE" ]; do
  pgrep -f run_table1_rest >/dev/null || { echo "table1rest gone without sentinel; continuing anyway" | tee -a "$LOG"; break; }
  sleep 60
done
echo "starting ($(date))" | tee -a "$LOG"

run() {
  local id="$1"; shift
  if [ -f "$R/$id/metrics.json" ]; then
    echo "[skip] $id" | tee -a "$LOG"; return 0
  fi
  echo "[run ] $id" | tee -a "$LOG"
  python3 -m experiments.run_experiment --exp-id "$id" "$@" >>"$LOG" 2>&1 \
    || echo "[FAIL] $id" | tee -a "$LOG"
}

tgt() { case "$1" in cicids2017) echo 0.60 ;; nbaiot) echo 0.68 ;; botiot) echo 0.62 ;; esac; }
tgt_depth() { case "$1" in cicids2017) echo 0.50 ;; nbaiot) echo 0.60 ;; botiot) echo 0.55 ;; esac; }

# ---- Table 3: communication vs accuracy (LAN, alpha=0.5, 3 schemes) --------
echo "=== TABLE 3: compression ===" | tee -a "$LOG"
for DS in cicids2017 nbaiot botiot; do
  T=$(tgt "$DS")
  for COMP in none int8 topk; do
    for SEED in 0 1 2; do
      for S in sync fedasync fedbuff staleness; do
        run "t3/${DS}__${S}__${COMP}__s${SEED}" \
          --dataset "$DS" --part-tag "a0.5_${SEED}" --strategy "$S" \
          --hardware hetero --network lan --compression "$COMP" --target-f1 "$T"
      done
    done
  done
done

# ---- Table 4: small vs large model, mixed network --------------------------
echo "=== TABLE 4: heavy-model induction ===" | tee -a "$LOG"
for DS in cicids2017 nbaiot botiot; do
  T=$(tgt_depth "$DS")
  for SIZE in small large; do
    for SEED in 0 1 2; do
      for S in sync staleness; do
        run "t4/${DS}__${S}__${SIZE}__s${SEED}" \
          --dataset "$DS" --part-tag "a0.5_${SEED}" --strategy "$S" \
          --hardware hetero --network mixed --compression none \
          --model-size "$SIZE" --target-f1 "$T" --max-seconds 600
      done
    done
  done
done

touch "$R/t3t4.DONE"
echo "=== ALL DONE ($(date)) ===" | tee -a "$LOG"
