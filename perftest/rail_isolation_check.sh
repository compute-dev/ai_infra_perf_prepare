#!/bin/bash
# ============================================================================
# rail_isolation_check.sh — determine whether the fabric is RAIL-ISOLATED
#                           (RDMA works within a rail, fails across rails).
#
# Why this matters: many rail-optimised fabrics give each rail its own subnet
# and only allow RDMA inside a rail. That single property decides how you run
# everything afterwards:
#   - bandwidth may only be compared between nodes on the SAME rail/leaf
#   - cross-node pipeline parallelism can fail, because NCCL may pair the two
#     ends' HCAs across rails and the QP never establishes
#
# The verdict is one line: same-rail runs at full rate AND cross-rail fails to
# connect => rail isolated.
#
# ★ Do not just ping. The IP layer IS reachable across rails and will mislead
#   you; it is the RoCE path that is blocked.
#
# Usage:
#     bash rail_isolation_check.sh <nodeA> <nodeB> [devA] [devB] [gid]
#
#   devA, devB  the device each side uses for the same-rail probe
#               (default: the first ACTIVE device on each node)
#   gid         RoCE GID index (default: auto-detected)
#
# Options:
#     --cross-dev DEV   force the peer device used for the cross-rail probe
#     --ssh-port N      host ssh port (default 22)
#     --ssh-user USER   default root
#
# Example:
#     bash rail_isolation_check.sh node01 node02 mlx5_1 mlx5_1 3
#
# Requires perftest (ib_send_bw) on both nodes and memlock unlimited.
# ============================================================================
set -uo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

A="$1"; B="$2"; shift 2

# Optional positionals: devA devB gid
DEVA=""; DEVB=""; GID=""
if [ $# -ge 1 ]; then DEVA="$1"; shift; fi
if [ $# -ge 1 ]; then DEVB="$1"; shift; fi
if [ $# -ge 1 ]; then GID="$1";  shift; fi

CROSS_DEV=""
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-root}"
while [ $# -gt 0 ]; do
    case "$1" in
        --cross-dev) CROSS_DEV="${2:-}"; shift 2 ;;
        --ssh-port)  SSH_PORT="${2:-}"; shift 2 ;;
        --ssh-user)  SSH_USER="${2:-}"; shift 2 ;;
        -h|--help)   sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown option: $1 (use --help)" >&2; exit 1 ;;
    esac
done

SSHX="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p ${SSH_PORT}"
_ssh(){ ssh ${SSHX} "${SSH_USER}@$1" "$2" 2>&1; }

# ---- helpers on a node -----------------------------------------------------
_active_devices() {
    _ssh "$1" "for d in /sys/class/infiniband/*; do \
                 [ -e \"\$d/ports/1/state\" ] || continue; \
                 [ \"\$(cat \$d/ports/1/state 2>/dev/null)\" = '4: ACTIVE' ] && basename \$d; \
               done | sort -V"
}
_dev_ip() {   # IP on that device's netdev — the TCP control channel address
    _ssh "$1" "nd=\$(ls /sys/class/infiniband/$2/device/net 2>/dev/null | head -1); \
               [ -n \"\$nd\" ] && ip -4 -o addr show \$nd 2>/dev/null | awk '{print \$4}' | cut -d/ -f1 | head -1"
}

# ---- resolve devices -------------------------------------------------------
mapfile -t A_DEVS < <(_active_devices "$A")
mapfile -t B_DEVS < <(_active_devices "$B")
if [ "${#A_DEVS[@]}" -eq 0 ] || [ "${#B_DEVS[@]}" -eq 0 ]; then
    echo "[ERROR] no ACTIVE RDMA device on ${A} (${#A_DEVS[@]}) or ${B} (${#B_DEVS[@]})" >&2
    exit 1
fi
[ -z "${DEVA}" ] && DEVA="${A_DEVS[0]}"
[ -z "${DEVB}" ] && DEVB="${B_DEVS[0]}"

# GID: only meaningful on RoCE
if [ -z "${GID}" ]; then
    LINK_LAYER="$(_ssh "$A" "cat /sys/class/infiniband/${DEVA}/ports/1/link_layer 2>/dev/null")"
    if [ "${LINK_LAYER}" = "Ethernet" ]; then
        GID_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tools/detect_roce_gid.sh"
        GID="$(_ssh "$A" "bash -s -- ${DEVA} < ${GID_HELPER}" | tail -1)"
        [[ "${GID}" =~ ^[0-9]+$ ]] || { echo "[ERROR] could not detect the RoCE GID index; pass it as the 5th argument" >&2; exit 1; }
    else
        GID="0"
    fi
fi

# ---- pick the cross-rail device on B --------------------------------------
# A different device on B means a different rail. Prefer the one furthest in
# order from DEVB, which is the most demanding pairing.
if [ -z "${CROSS_DEV}" ]; then
    BEST_DIST=-1
    for d in "${B_DEVS[@]}"; do
        [ "${d}" = "${DEVB}" ] && continue
        # distance by trailing number where possible, else count position
        n_d="${d##*_}"; n_b="${DEVB##*_}"
        if [[ "${n_d}" =~ ^[0-9]+$ ]] && [[ "${n_b}" =~ ^[0-9]+$ ]]; then
            dist=$(( n_d > n_b ? n_d - n_b : n_b - n_d ))
        else
            dist=1
        fi
        if [ "${dist}" -gt "${BEST_DIST}" ]; then BEST_DIST="${dist}"; CROSS_DEV="${d}"; fi
    done
    if [ -z "${CROSS_DEV}" ]; then
        echo "[ERROR] ${B} exposes only one ACTIVE device (${DEVB}); cannot form a cross-rail case." >&2
        echo "        A rail-isolation check needs at least two devices on the peer." >&2
        exit 1
    fi
fi

IP_SAME="$(_dev_ip "$B" "$DEVB")"
IP_CROSS="$(_dev_ip "$B" "$CROSS_DEV")"
[ -n "${IP_SAME}" ]  || { echo "[ERROR] no IPv4 on ${B}:${DEVB} — is the rail cabled and addressed?" >&2; exit 1; }
[ -n "${IP_CROSS}" ] || { echo "[ERROR] no IPv4 on ${B}:${CROSS_DEV}" >&2; exit 1; }

echo "===== rail isolation check: ${A}(${DEVA}) -> ${B}(${DEVB}) ====="
echo "  same rail : client uses ${DEVA}, peer uses ${DEVB} (peer IP ${IP_SAME})"
echo "  cross rail: client uses ${DEVA}, peer uses ${CROSS_DEV} (peer IP ${IP_CROSS})"
echo "  Note: which rail RDMA takes is decided by EACH END'S -d device. The IP is"
echo "        only the TCP control channel, so crossing rails requires changing the"
echo "        peer's DEVICE. Changing only the IP silently stays on the same rail."

# _probe <label> <peer-device> <dest-ip>
_probe() {
    _ssh "$B" "pkill -x ib_send_bw 2>/dev/null; sleep 1; (setsid nohup timeout 25 ib_send_bw -d $2 -x ${GID} -F --report_gbits >/tmp/railchk_$1.txt 2>&1 < /dev/null &)" >/dev/null 2>&1
    sleep 3

    # Verify the peer server actually started. With a mistyped device name the
    # server never comes up, the client reports "couldn't connect", and that
    # would be misread as "rail isolated". Refuse to judge in that case.
    local srv
    srv="$(_ssh "$B" "head -20 /tmp/railchk_$1.txt 2>/dev/null")"
    if echo "${srv}" | grep -qiE "not found|Unable to find|command not found"; then
        echo "  [FATAL] the peer server did not start (device or environment problem)." >&2
        echo "          This probe is invalid:" >&2
        echo "${srv}" | grep -iE "not found|Unable to find" | head -2 | sed 's/^/    /' >&2
        return 1
    fi

    _ssh "$A" "timeout 20 ib_send_bw -d ${DEVA} -x ${GID} -F --report_gbits $3 2>&1 | tail -6"
}

echo
echo "--- same rail ---"
SAME="$(_probe same "${DEVB}" "${IP_SAME}")" || { echo "[ABORT] the same-rail baseline did not run; cannot judge. Check device name / GID / cabling." ; exit 1; }
echo "${SAME}" | sed 's/^/  /'

echo "--- cross rail ---"
CROSS="$(_probe cross "${CROSS_DEV}" "${IP_CROSS}" || true)"
echo "${CROSS}" | sed 's/^/  /'

# Self-check: the cross-rail run's remote GID should point at the OTHER rail.
# If it still shows the same-rail address, the peer device did not actually
# change and the conclusion is untrustworthy.
if echo "${CROSS}" | grep -q "$(echo "${IP_SAME}" | tr '.' ':')"; then
    echo
    echo "  [WARN] the cross-rail run's remote GID still points at the same rail —"
    echo "         the peer device did not change. This result is NOT trustworthy;"
    echo "         check that ${CROSS_DEV} really belongs to a different rail."
fi

echo
echo "===== verdict ====="
same_bw="$(echo "${SAME}"  | awk '/^ *65536/{print $4}' | tail -1)"
cross_bw="$(echo "${CROSS}" | awk '/^ *65536/{print $4}' | tail -1)"
echo "  same-rail BW = ${same_bw:-<none>} Gb/s ; cross-rail BW = ${cross_bw:-<none>} Gb/s"

if [ -n "${same_bw}" ] && [ -z "${cross_bw}" ]; then
    echo "  ** RAIL ISOLATED ** same-rail runs at full rate, cross-rail fails to connect."
    echo "     Consequences:"
    echo "       1. Only compare bandwidth between nodes on the same rail/leaf."
    echo "       2. Cross-node pipeline parallelism is sensitive to rail pairing: if"
    echo "          NCCL pairs the two ends' HCAs across rails the QP fails with"
    echo "          'ibv_modify_qp INIT->RTR ETIMEDOUT' — and only for SOME node pairs"
    echo "          (measured: 6 of 12 pairs worked, and it was not a fixed set of nodes)."
    echo "       3. Locating leaf boundaries needs the network side to give you an"
    echo "          'HCA port -> leaf/rail' map; the host cannot see leaf from here."
    echo "     Workaround: keep pipeline parallelism inside a node (use tp/dp across"
    echo "     nodes), or probe node pairs with a small pp>1 run first."
elif [ -n "${same_bw}" ] && [ -n "${cross_bw}" ]; then
    echo "  NOT rail isolated (cross-rail also establishes) — grouping is unconstrained;"
    echo "  judge by leaf oversubscription instead."
else
    echo "  The same-rail case itself did not run — check device name / GID / cabling"
    echo "  first. This is not a rail-isolation result."
fi
