set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/close2.log
: > "$LOG"
run(){ id="$1"; shift; [ -f "$R/$id/metrics.json" ] && { echo "[skip] $id" >>"$LOG"; return 0; }
  echo "[run ] $id $(date +%H:%M:%S)" >>"$LOG"
  python3 -m experiments.run_experiment --exp-id "$id" "$@" >>"$LOG" 2>&1 || echo "[FAIL] $id" >>"$LOG"; }

# A) buffered-NS ablation on the corrected harness, equal 450 s budget,
#    same protocol as finsweep so it slots into Table 2. Strategy innermost.
for A in a0.1 a0.5; do for SEED in 0 1 2; do for S in fedbuff fedbuff_ns staleness; do
  run "fin_ns/cicids2017__${S}__${A}__s${SEED}" --dataset cicids2017 \
    --part-tag "${A}_${SEED}" --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.99 --max-seconds 450 --max-version 100000
done; done; done

# B) local work per push: the fourth manipulation of the per-round burden.
for STEPS in 10 30 60 120; do for SEED in 0 1 2; do for S in sync staleness; do
  run "fin_steps/cicids2017__${S}__n${STEPS}__s${SEED}" --dataset cicids2017 \
    --part-tag a0.5_$SEED --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --local-steps $STEPS --max-seconds 900
done; done; done
touch "$R/close2.DONE"; echo "=== DONE ($(date)) ===" >>"$LOG"
