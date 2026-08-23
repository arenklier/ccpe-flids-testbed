set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/skew.log
: > "$LOG"

run(){ id="$1"; shift; [ -f "$R/$id/metrics.json" ] && { echo "[skip] $id" >>"$LOG"; return 0; }
  echo "[run ] $id $(date +%H:%M:%S)" >>"$LOG"
  python3 -m experiments.run_experiment --exp-id "$id" "$@" >>"$LOG" 2>&1 || echo "[FAIL] $id" >>"$LOG"; }

# Hold the dataset, the label skew and the total number of training rows fixed
# and vary only how unevenly those rows are spread across the twelve clients.
echo "=== building re-skewed partitions ===" >>"$LOG"
for RT in 1 5 25 100; do for SEED in 0 1 2; do
  D=datasets/prepared/cicids2017_skew${RT}/a0.5_${SEED}
  [ -f "$D/meta.json" ] && { echo "[skip] part skew$RT s$SEED" >>"$LOG"; continue; }
  docker run --rm -v /mnt/data/ccpe-flids:/work -w /work ccpe-flids:latest \
    python experiments/reskew.py --src datasets/prepared/cicids2017/a0.5_${SEED} \
    --out "$D" --ratio $RT --seed $SEED >>"$LOG" 2>&1 \
    || echo "[FAIL] part skew$RT s$SEED" >>"$LOG"
done; done

echo "=== buffer-size sweep at four shard-size ratios ===" >>"$LOG"
for RT in 1 5 25 100; do for K in 2 4 8 16; do for SEED in 0 1 2; do for S in fedbuff staleness; do
  run "fin_skew/r${RT}__${S}__K${K}__s${SEED}" --dataset cicids2017_skew${RT} \
    --part-tag a0.5_$SEED --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --buffer-k $K --max-seconds 600
done; done; done; done

touch "$R/skew.DONE"; echo "=== DONE ($(date)) ===" >>"$LOG"
