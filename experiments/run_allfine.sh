set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/allfine.log
: > "$LOG"

# Every batch behind a table or figure, re-measured with the stratified
# checkpoint set so that one evaluator is used throughout the paper. Strategy
# stays the innermost loop everywhere, so compared arms remain adjacent.
FINE="--eval-every 1 --eval-cap 5000"

run(){ id="$1"; shift; [ -f "$R/$id/metrics.json" ] && { echo "[skip] $id" >>"$LOG"; return 0; }
  echo "[run ] $id $(date +%H:%M:%S)" >>"$LOG"
  python3 -m experiments.run_experiment --exp-id "$id" $FINE "$@" >>"$LOG" 2>&1 \
    || echo "[FAIL] $id" >>"$LOG"; }
tgt(){ case "$1" in cicids2017) echo 0.60;; nbaiot) echo 0.68;; botiot) echo 0.62;; esac; }
tgtd(){ case "$1" in cicids2017) echo 0.50;; nbaiot) echo 0.60;; botiot) echo 0.55;; esac; }

echo "=== equal-budget quality (Table 2) ===" >>"$LOG"
for A in a0.1 a0.5; do for SEED in 0 1 2; do for S in sync fedasync fedbuff staleness; do
  run "finsweep_f/cicids2017__${S}__${A}__s${SEED}" --dataset cicids2017 \
    --part-tag "${A}_${SEED}" --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.99 --max-seconds 450 --max-version 100000
done; done; done

echo "=== communication (Table 3) ===" >>"$LOG"
for DS in cicids2017 nbaiot botiot; do T=$(tgt $DS)
for COMP in none int8 topk; do for SEED in 0 1 2; do for S in sync fedasync fedbuff staleness; do
  run "t3_f/${DS}__${S}__${COMP}__s${SEED}" --dataset $DS --part-tag a0.5_$SEED \
    --strategy $S --hardware hetero --network lan --compression $COMP --target-f1 $T
done; done; done; done

echo "=== local work (Table 4) ===" >>"$LOG"
for STEPS in 10 30 60 120; do for SEED in 0 1 2; do for S in sync staleness; do
  run "fin_steps_f/cicids2017__${S}__n${STEPS}__s${SEED}" --dataset cicids2017 \
    --part-tag a0.5_$SEED --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --local-steps $STEPS --max-seconds 900
done; done; done

echo "=== model size (Table 5) ===" >>"$LOG"
for DS in cicids2017 nbaiot botiot; do T=$(tgtd $DS)
for SIZE in small large; do for SEED in 0 1 2; do for S in sync staleness; do
  run "fin_depth_f/${DS}__${S}__${SIZE}__s${SEED}" --dataset $DS --part-tag a0.5_$SEED \
    --strategy $S --hardware hetero --network mixed --compression none \
    --model-size $SIZE --target-f1 $T --max-seconds 600
done; done; done; done

echo "=== client count (Table 6) ===" >>"$LOG"
for N in 12 24 48 96; do
  DS=cicids2017; [ $N -gt 12 ] && DS=cicids2017_nc$N
  for SEED in 0 1 2; do for S in sync fedasync fedbuff staleness; do
    run "finscale_f/n${N}__${S}__s${SEED}" --dataset $DS --part-tag a0.5_$SEED \
      --strategy $S --hardware homo --network lan --compression none \
      --n-clients $N --target-f1 0.60 --max-seconds 900
  done; done
done

echo "=== shard-size skew (Table 7) ===" >>"$LOG"
for RT in 1 5 25 100; do for K in 2 4 8 16; do for SEED in 0 1 2; do for S in fedbuff staleness; do
  run "fin_skew_f/r${RT}__${S}__K${K}__s${SEED}" --dataset cicids2017_skew${RT} \
    --part-tag a0.5_$SEED --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --buffer-k $K --max-seconds 600
done; done; done; done

echo "=== delay curve (Figure 2) ===" >>"$LOG"
for D in 1 25 50 100 150 175 200 225 250; do for SEED in 0 1 2; do for S in sync staleness; do
  run "fin_delay_f/cicids2017__${S}__d${D}__s${SEED}" --dataset cicids2017 \
    --part-tag a0.5_$SEED --strategy $S --hardware hetero --network $D \
    --compression none --target-f1 0.60 --max-seconds 400
done; done; done

echo "=== label skew (Figure 3) ===" >>"$LOG"
for TAG in a0.1 a0.5 iid; do for SEED in 0 1 2; do for S in sync fedasync fedbuff staleness; do
  run "fin_noniid_f/cicids2017__${S}__${TAG}__s${SEED}" --dataset cicids2017 \
    --part-tag "${TAG}_${SEED}" --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --max-seconds 900
done; done; done

echo "=== tier ratios ===" >>"$LOG"
for MIX in 2,2,8 8,2,2; do TAGM=$(echo $MIX | tr ',' '-')
for SEED in 0 1 2; do for S in sync staleness; do
  run "fin_tier_f/cicids2017__${S}__${TAGM}__s${SEED}" --dataset cicids2017 \
    --part-tag a0.5_$SEED --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --tier-mix $MIX --max-seconds 900
done; done; done

echo "=== model size on a LAN ===" >>"$LOG"
for SIZE in small large; do for SEED in 0 1 2; do for S in sync staleness; do
  run "fin_depthlan_f/cicids2017__${S}__${SIZE}__s${SEED}" --dataset cicids2017 \
    --part-tag a0.5_$SEED --strategy $S --hardware hetero --network lan \
    --compression none --model-size $SIZE --target-f1 0.50 --max-seconds 600
done; done; done

echo "=== buffered-NS ablation ===" >>"$LOG"
for A in a0.1 a0.5; do for SEED in 0 1 2; do for S in fedbuff fedbuff_ns staleness; do
  run "fin_ns_f/cicids2017__${S}__${A}__s${SEED}" --dataset cicids2017 \
    --part-tag "${A}_${SEED}" --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.99 --max-seconds 450 --max-version 100000
done; done; done

echo "=== server rate and buffer size, CICIDS ===" >>"$LOG"
for ETA in 0.1 0.3 0.5 1.0; do for SEED in 0 1 2; do for S in fedbuff staleness; do
  run "finsens_eta_f/cicids2017__${S}__eta${ETA}__s${SEED}" --dataset cicids2017 \
    --part-tag a0.5_$SEED --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --server-lr-override $ETA --max-seconds 600
done; done; done
for K in 2 4 8 16; do for SEED in 0 1 2; do for S in fedbuff staleness; do
  run "finsens_k_f/cicids2017__${S}__K${K}__s${SEED}" --dataset cicids2017 \
    --part-tag a0.5_$SEED --strategy $S --hardware hetero --network mixed \
    --compression none --target-f1 0.60 --buffer-k $K --max-seconds 600
done; done; done

echo "=== the same sweeps on the other two datasets ===" >>"$LOG"
for DS in nbaiot botiot; do T=$(tgt $DS)
for D in 1 50 150 250; do for SEED in 0 1 2; do for S in sync staleness; do
  run "fin_delay2_f/${DS}__${S}__d${D}__s${SEED}" --dataset $DS --part-tag a0.5_$SEED \
    --strategy $S --hardware hetero --network $D --compression none \
    --target-f1 $T --max-seconds 600
done; done; done; done
for DS in nbaiot botiot; do T=$(tgt $DS)
for ETA in 0.1 0.3 0.5 1.0; do for SEED in 0 1 2; do for S in fedbuff staleness; do
  run "finsens_eta2_f/${DS}__${S}__eta${ETA}__s${SEED}" --dataset $DS --part-tag a0.5_$SEED \
    --strategy $S --hardware hetero --network mixed --compression none \
    --target-f1 $T --server-lr-override $ETA --max-seconds 600
done; done; done; done
for DS in nbaiot botiot; do T=$(tgt $DS)
for K in 2 4 8 16; do for SEED in 0 1 2; do for S in fedbuff staleness; do
  run "finsens_k2_f/${DS}__${S}__K${K}__s${SEED}" --dataset $DS --part-tag a0.5_$SEED \
    --strategy $S --hardware hetero --network mixed --compression none \
    --target-f1 $T --buffer-k $K --max-seconds 600
done; done; done; done

touch "$R/allfine.DONE"; echo "=== DONE ($(date)) ===" >>"$LOG"
