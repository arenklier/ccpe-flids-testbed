set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/lock.log
: > "$LOG"
FINE="--eval-every 1 --eval-cap 5000"

run(){ id="$1"; shift; [ -f "$R/$id/metrics.json" ] && { echo "[skip] $id" >>"$LOG"; return 0; }
  echo "[run ] $id $(date +%H:%M:%S)" >>"$LOG"
  python3 -m experiments.run_experiment --exp-id "$id" $FINE "$@" >>"$LOG" 2>&1 \
    || echo "[FAIL] $id" >>"$LOG"; }

# The barrier releases twelve clients at once; an asynchronous rule spreads
# them out. Measure the queueing this causes at the aggregator, in the two
# conditions where it should matter most: a fast LAN, where round times are
# short enough for contention to be a visible share, and the large model,
# where each aggregation does the most work inside the lock.
for NET in lan mixed; do for SIZE in small large; do for SEED in 0 1 2; do
  for S in sync fedasync; do
    T=0.60; [ "$SIZE" = "large" ] && T=0.50
    run "fin_lock/${S}__${NET}__${SIZE}__s${SEED}" --dataset cicids2017 \
      --part-tag a0.5_$SEED --strategy $S --hardware hetero --network $NET \
      --compression none --model-size $SIZE --target-f1 $T --max-seconds 600
  done
done; done; done

touch "$R/lock.DONE"; echo "=== DONE ($(date)) ===" >>"$LOG"
