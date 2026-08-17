#!/usr/bin/env bash
# Launch the real (non-containerised) clients for the cross-host validation.
#
# Run this on the second physical machine while the aggregator and the local
# docker clients are already up on the server. These clients get no cgroup
# limit and no netem shaping: they are the genuine article, reaching the
# aggregator over the institution's real network.
#
# Usage: bash experiments/real_clients.sh <server-ip> <first-rank> <n-clients>
set -uo pipefail

SERVER="${1:?usage: real_clients.sh <server-ip> <first-rank> <n-clients>}"
FIRST="${2:?missing first rank}"
N="${3:?missing client count}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

export PYTHONPATH="$HERE/src"
export SERVER_ADDR="${SERVER}:8080"
export PARTITION_DIR="${PARTITION_DIR:-$HERE/real_client_data}"
export IN_DIM=76
export N_CLASSES=12
export LOCAL_STEPS=30
export LOCAL_LR=0.05
export COMPRESSION=none
export MODEL_SIZE=small

pids=()
for ((i=0; i<N; i++)); do
  RANK=$((FIRST + i))
  CLIENT_ID=$(printf "c%02d" "$RANK") CLIENT_RANK=$RANK TIER=fast CPU_BUDGET=4 \
    python -m flids.client.main --tier fast > "/tmp/realclient_${RANK}.log" 2>&1 &
  pids+=($!)
  echo "started real client rank=$RANK pid=${pids[-1]}"
done

echo "waiting for ${#pids[@]} real clients..."
for pid in "${pids[@]}"; do wait "$pid"; done
echo "all real clients exited"
