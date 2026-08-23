set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/close3.log
: > "$LOG"

run(){ id="$1"; shift; [ -f "$R/$id/metrics.json" ] && { echo "[skip] $id" >>"$LOG"; return 0; }
  echo "[run ] $id $(date +%H:%M:%S)" >>"$LOG"
  python3 -m experiments.run_experiment --exp-id "$id" "$@" >>"$LOG" 2>&1 || echo "[FAIL] $id" >>"$LOG"; }

# ---------------------------------------------------------------- (b) seeds
# The client-count sweep was single-seed; build the missing partitions for the
# larger fleets, then repeat the whole sweep at seeds 1 and 2.
echo "=== preparing partitions for the larger fleets ===" >>"$LOG"
for N in 24 48 96; do for SEED in 1 2; do
  D=datasets/prepared/cicids2017_nc${N}/a0.5_${SEED}
  [ -d "$D" ] && { echo "[skip] part nc$N s$SEED" >>"$LOG"; continue; }
  echo "[prep] nc$N s$SEED" >>"$LOG"
  docker run --rm -v /mnt/data/ccpe-flids:/work -w /work ccpe-flids:latest \
    python -m experiments.prepare_data --dataset cicids2017 --raw datasets/cicids2017 \
    --out datasets/prepared/cicids2017_nc${N} --n-clients $N --alpha 0.5 --seed $SEED \
    >>"$LOG" 2>&1 || echo "[FAIL] prep nc$N s$SEED" >>"$LOG"
done; done

echo "=== (b) client-count sweep, seeds 1 and 2 ===" >>"$LOG"
for N in 12 24 48 96; do
  DS=cicids2017; [ $N -gt 12 ] && DS=cicids2017_nc$N
  for SEED in 1 2; do for S in sync fedasync fedbuff staleness; do
    run "finscale/n${N}__${S}__s${SEED}" --dataset $DS --part-tag a0.5_$SEED \
      --strategy $S --hardware homo --network lan --compression none \
      --n-clients $N --target-f1 0.60 --max-seconds 900
  done; done
done

# ------------------------------------------------------- (c) sweeps off CICIDS
# Delay curve, four representative points, on the two datasets the paper only
# ever swept at two points.
echo "=== (c) delay curve on N-BaIoT and Bot-IoT ===" >>"$LOG"
for DS in nbaiot botiot; do
  TGT=0.68; [ "$DS" = "botiot" ] && TGT=0.62
  for D in 1 50 150 250; do for SEED in 0 1 2; do for S in sync staleness; do
    run "fin_delay2/${DS}__${S}__d${D}__s${SEED}" --dataset $DS --part-tag a0.5_$SEED \
      --strategy $S --hardware hetero --network $D --compression none \
      --target-f1 $TGT --max-seconds 600
  done; done; done
done

echo "=== (c) server learning rate on N-BaIoT and Bot-IoT ===" >>"$LOG"
for DS in nbaiot botiot; do
  TGT=0.68; [ "$DS" = "botiot" ] && TGT=0.62
  for ETA in 0.1 0.3 0.5 1.0; do for SEED in 0 1 2; do for S in fedbuff staleness; do
    run "finsens_eta2/${DS}__${S}__eta${ETA}__s${SEED}" --dataset $DS --part-tag a0.5_$SEED \
      --strategy $S --hardware hetero --network mixed --compression none \
      --target-f1 $TGT --server-lr-override $ETA --max-seconds 600
  done; done; done
done

echo "=== (c) buffer size on N-BaIoT and Bot-IoT ===" >>"$LOG"
for DS in nbaiot botiot; do
  TGT=0.68; [ "$DS" = "botiot" ] && TGT=0.62
  for K in 2 4 8 16; do for SEED in 0 1 2; do for S in fedbuff staleness; do
    run "finsens_k2/${DS}__${S}__K${K}__s${SEED}" --dataset $DS --part-tag a0.5_$SEED \
      --strategy $S --hardware hetero --network mixed --compression none \
      --target-f1 $TGT --buffer-k $K --max-seconds 600
  done; done; done
done

touch "$R/close3.DONE"; echo "=== DONE ($(date)) ===" >>"$LOG"
