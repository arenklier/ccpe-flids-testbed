set -uo pipefail
cd /mnt/data/ccpe-flids
R=/mnt/data/ccpe-flids/results
LOG=$R/fine.log
: > "$LOG"
rm -rf "$R/probe"

run(){ id="$1"; shift; [ -f "$R/$id/metrics.json" ] && { echo "[skip] $id" >>"$LOG"; return 0; }
  echo "[run ] $id $(date +%H:%M:%S)" >>"$LOG"
  python3 -m experiments.run_experiment --exp-id "$id" "$@" >>"$LOG" 2>&1 || echo "[FAIL] $id" >>"$LOG"; }

# Table 1, re-measured with a checkpoint cheap enough to take often. The
# stratified set is reweighted by the inverse class sampling rate, so the
# curve is on the same scale as a full pass; each run also records both.
for DS in cicids2017 nbaiot botiot; do
  T=0.60; [ "$DS" = "nbaiot" ] && T=0.68; [ "$DS" = "botiot" ] && T=0.62
  for NET in lan mixed; do for SEED in 0 1 2; do
    for S in sync fedasync fedbuff staleness; do
      run "fin_cross2/${DS}__${S}__${NET}__s${SEED}" --dataset $DS \
        --part-tag a0.5_$SEED --strategy $S --hardware hetero --network $NET \
        --compression none --target-f1 $T --eval-every 1 --eval-cap 5000 \
        --max-seconds 900
    done
  done; done
done
touch "$R/fine.DONE"; echo "=== DONE ($(date)) ===" >>"$LOG"
