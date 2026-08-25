set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/beta.log
: > "$LOG"
FINE="--eval-every 1 --eval-cap 5000"

run(){ id="$1"; shift; [ -f "$R/$id/metrics.json" ] && { echo "[skip] $id" >>"$LOG"; return 0; }
  echo "[run ] $id $(date +%H:%M:%S)" >>"$LOG"
  python3 -m experiments.run_experiment --exp-id "$id" $FINE "$@" >>"$LOG" 2>&1 \
    || echo "[FAIL] $id" >>"$LOG"; }

# Does compressing the range of shard sizes recover what the data-size term was
# supposed to buy? beta = 0 is FedBuff exactly, beta = 1 the plain data-size
# rule. Run on the re-skewed partitions so the skew is a controlled quantity.
echo "=== exponent sweep on skewed partitions ===" >>"$LOG"
for RT in 25 100; do for B in 0 0.25 0.5 0.75 1; do for K in 4 16; do for SEED in 0 1 2; do
  run "fin_beta/r${RT}__b${B}__K${K}__s${SEED}" --dataset cicids2017_skew${RT} \
    --part-tag a0.5_$SEED --strategy staleness --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --buffer-k $K --size-beta $B --max-seconds 600
done; done; done; done

echo "=== and where there is no skew to compress ===" >>"$LOG"
for B in 0 0.5 1; do for K in 4 16; do for SEED in 0 1 2; do
  run "fin_beta/r1__b${B}__K${K}__s${SEED}" --dataset cicids2017_skew1 \
    --part-tag a0.5_$SEED --strategy staleness --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --buffer-k $K --size-beta $B --max-seconds 600
done; done; done

echo "=== median clipping, the other way to cap a large shard ===" >>"$LOG"
for RT in 1 25 100; do for K in 4 16; do for SEED in 0 1 2; do
  run "fin_clip/r${RT}__K${K}__s${SEED}" --dataset cicids2017_skew${RT} \
    --part-tag a0.5_$SEED --strategy staleness --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --buffer-k $K --size-clip --max-seconds 600
done; done; done

touch "$R/beta.DONE"; echo "=== DONE ($(date)) ===" >>"$LOG"
