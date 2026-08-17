#!/usr/bin/env bash
# Cross-host validation, driven from the second physical machine.
#
# For each (strategy, seed) this starts the aggregator and eight containerised
# clients on the server, waits for the aggregator's port to accept
# connections, then starts four real client processes here. Those four are not
# containerised, not cgroup-limited and not netem-shaped; they reach the
# aggregator over the institution's real network. Runs are strictly
# sequential because wall-clock is the metric.
#
# Usage: bash experiments/run_crosshost.sh
set -uo pipefail

SERVER_IP=172.30.0.209
SERVER_USER=ayhan
REMOTE=/mnt/data/ccpe-flids
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$HERE/crosshost_local.log"
: > "$LOG"

for SEED in 0 1 2; do
  for S in sync fedasync fedbuff staleness; do
    ID="xh/${S}__s${SEED}"
    if ssh "$SERVER_USER@$SERVER_IP" "test -f $REMOTE/results/$ID/metrics.json" 2>/dev/null; then
      echo "[skip] $ID" | tee -a "$LOG"; continue
    fi
    echo "[run ] $ID" | tee -a "$LOG"

    ssh "$SERVER_USER@$SERVER_IP" "cd $REMOTE && (setsid nohup python3 -m experiments.run_experiment \
      --exp-id $ID --dataset cicids2017 --part-tag a0.5_${SEED} --strategy $S \
      --hardware hetero --network lan --compression none --target-f1 0.60 \
      --n-clients 12 --docker-clients 8 --publish-port \
      > /tmp/xh_${S}_${SEED}.log 2>&1 </dev/null &)" >>"$LOG" 2>&1

    # wait for the aggregator to accept connections before attaching
    for _ in $(seq 1 60); do
      if python -c "import socket,sys; s=socket.socket(); s.settimeout(1);
sys.exit(0 if s.connect_ex(('$SERVER_IP',8080))==0 else 1)" 2>/dev/null; then
        break
      fi
      sleep 2
    done

    PARTITION_DIR="$HERE/real_client_data/a0.5_${SEED}" \
      timeout 900 bash "$HERE/experiments/real_clients.sh" "$SERVER_IP" 8 4 >>"$LOG" 2>&1

    # let the server finish writing metrics before the next run
    for _ in $(seq 1 60); do
      ssh "$SERVER_USER@$SERVER_IP" "test -f $REMOTE/results/$ID/metrics.json" 2>/dev/null && break
      sleep 3
    done
    echo "[done] $ID" | tee -a "$LOG"
  done
done

echo "=== cross-host validation complete ===" | tee -a "$LOG"
