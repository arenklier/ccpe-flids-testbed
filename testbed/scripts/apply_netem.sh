#!/usr/bin/env bash
# Apply tc netem to a run's client containers, by hardware tier.
# Usage: apply_netem.sh <lan|mixed|NUMBER> <run-prefix>
#   run-prefix is the docker name prefix; clients are <prefix>-c<NN>.
# Tier is read from each container's TIER env (fast|mid|slow).
#   lan     : all clients 1ms, unlimited          (network=lan baseline)
#   mixed   : fast 1ms | mid 50ms,10mbit | slow 150ms,2mbit,1% loss (fixed point)
#   NUMBER  : continuous severity knob = target slow-tier delay in ms (D).
#             fast stays 1ms; mid = D/3 ms, 10mbit; slow = D ms, 2mbit,
#             loss = min(1%, D/150 * 1%). Lets the crossover be traced as a
#             function of one scalar network-heterogeneity magnitude.
# No host root needed: clients are started with --cap-add NET_ADMIN and the
# image ships iproute2, so tc runs inside each container via docker exec.
set -euo pipefail

PROFILE="${1:?usage: apply_netem.sh <lan|mixed|NUMBER> <run-prefix>}"
PREFIX="${2:?missing run-prefix}"

shape() {
  local ctr="$1" delay="$2" rate="$3" loss="$4"
  docker exec "$ctr" tc qdisc replace dev eth0 root netem \
    delay "$delay" ${rate:+rate "$rate"} ${loss:+loss "$loss"}
  echo "netem $ctr (${5:-?}) -> delay=$delay rate=${rate:-none} loss=${loss:-0%}"
}

mapfile -t ctrs < <(docker ps --format '{{.Names}}' | grep -E "^${PREFIX}-c[0-9]+$")
[ "${#ctrs[@]}" -gt 0 ] || { echo "apply_netem: no containers match ${PREFIX}-c*" >&2; exit 1; }

for ctr in "${ctrs[@]}"; do
  tier=$(docker inspect "$ctr" --format '{{range .Config.Env}}{{println .}}{{end}}' \
         | sed -n 's/^TIER=//p')
  case "$PROFILE" in
    lan) shape "$ctr" 1ms "" "" "$tier" ;;
    mixed)
      case "$tier" in
        fast) shape "$ctr" 1ms   ""      ""  "$tier" ;;
        mid)  shape "$ctr" 50ms  10mbit  ""  "$tier" ;;
        slow) shape "$ctr" 150ms 2mbit   1%  "$tier" ;;
        *)    echo "unknown tier for $ctr" >&2 ;;
      esac ;;
    ''|*[!0-9]*) echo "unknown profile: $PROFILE" >&2; exit 1 ;;
    *)
      D="$PROFILE"
      mid_d=$(( D / 3 )); [ "$mid_d" -lt 1 ] && mid_d=1
      loss_pct=$(awk -v d="$D" 'BEGIN{p=d/150.0; if(p>1)p=1; printf "%.2f", p}')
      case "$tier" in
        fast) shape "$ctr" 1ms       ""      ""            "$tier" ;;
        mid)  shape "$ctr" "${mid_d}ms" 10mbit ""           "$tier" ;;
        slow) shape "$ctr" "${D}ms"  2mbit   "${loss_pct}%" "$tier" ;;
        *)    echo "unknown tier for $ctr" >&2 ;;
      esac ;;
  esac
done
