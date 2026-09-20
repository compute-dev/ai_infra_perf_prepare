#!/bin/bash
# ============================================================================
# diag_gid.sh — diagnose the RoCE GID index between two nodes.
#
# The GID index is the single most common cause of "the fabric looks broken but
# isn't": a wrong index makes every queue pair fail to establish with
#     ibv_modify_qp INIT->RTR ETIMEDOUT
# on both InfiniBand and RoCE fabrics it reads like a cabling or switch fault.
#
# The index is assigned by the driver from the port's GID table. It shifts when
# the IP scheme, the RoCE version (v1 vs v2), or the driver changes. That is why
# hardcoding 3 across clusters eventually breaks, and why this script exists.
#
# What it does:
#   1. dumps the GID table for the device on both nodes
#   2. auto-detects the RoCE v2 + IPv4 index on each node
#   3. flags a mismatch between the two nodes
#   4. re-runs ib_write_bw with the detected index to confirm it works
#
# Usage:
#     bash diag_gid.sh <nodeA> <nodeB> [dev]
#
# Options:
#     --ssh-port N     host ssh port (default 22)
#     --ssh-user USER  default root
#     --no-test        only dump and detect, do not run a bandwidth test
#
# Requires perftest (ib_write_bw) on both nodes for the confirmation step.
# ============================================================================
set -uo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

A="$1"; B="$2"; shift 2
DEV="${1:-}"; [ $# -ge 1 ] && shift

SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-root}"
NO_TEST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --ssh-port)  SSH_PORT="${2:-}"; shift 2 ;;
        --ssh-user)  SSH_USER="${2:-}"; shift 2 ;;
        --no-test)   NO_TEST=1; shift ;;
        -h|--help)   sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown option: $1 (use --help)" >&2; exit 1 ;;
    esac
done

SSHX="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p ${SSH_PORT}"
_ssh(){ ssh ${SSHX} "${SSH_USER}@$1" "$2" 2>&1; }

# _ssh_script <node> <local_script> [args...] — run a LOCAL script on a remote
# node via stdin. The redirection must be local: inside the remote command
# string the remote shell would look for a path that only exists here.
_ssh_script() {
    local node="$1" script="$2"; shift 2
    ssh ${SSHX} "${SSH_USER}@${node}" "bash -s -- $*" < "${script}" 2>&1
}

GID_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tools/detect_roce_gid.sh"
if [ ! -f "${GID_HELPER}" ]; then
    echo "[ERROR] ${GID_HELPER} not found (needed to run on the remote nodes)" >&2
    exit 1
fi

# Default device: the first ACTIVE one on A
if [ -z "${DEV}" ]; then
    DEV="$(_ssh "$A" "for d in /sys/class/infiniband/*; do \
        [ -e \"\$d/ports/1/state\" ] || continue; \
        [ \"\$(cat \$d/ports/1/state 2>/dev/null)\" = '4: ACTIVE' ] && basename \$d; \
      done | sort -V | head -1")"
    if [ -z "${DEV}" ]; then
        echo "[ERROR] no ACTIVE RDMA device on ${A}" >&2
        exit 1
    fi
fi

echo "===== GID diagnostic: ${A} <-> ${B}  device=${DEV} ====="

LINK_LAYER="$(_ssh "$A" "cat /sys/class/infiniband/${DEV}/ports/1/link_layer 2>/dev/null")"
echo "[INFO] link layer: ${LINK_LAYER:-unknown}"

if [ "${LINK_LAYER}" = "InfiniBand" ]; then
    echo
    echo "This is an InfiniBand fabric. GID indices do not apply — leave"
    echo "NCCL_IB_GID_INDEX unset and look elsewhere for the fault."
    exit 0
fi

# ---- dump the GID table on both nodes -------------------------------------
for node in "${A}" "${B}"; do
    echo
    echo "--- ${node}:${DEV} GID table ---"
    _ssh_script "${node}" "${GID_HELPER}" "${DEV}" --table | sed 's/^/  /'
done

# ---- detect the index on each node ----------------------------------------
echo
IDX_A="$(_ssh_script "${A}" "${GID_HELPER}" "${DEV}" | tail -1)"
IDX_B="$(_ssh_script "${B}" "${GID_HELPER}" "${DEV}" | tail -1)"

echo "detected RoCE v2 + IPv4 GID index:"
echo "  ${A}: ${IDX_A:-<none>}"
echo "  ${B}: ${IDX_B:-<none>}"

if ! [[ "${IDX_A}" =~ ^[0-9]+$ ]] || ! [[ "${IDX_B}" =~ ^[0-9]+$ ]]; then
    echo
    echo "[ERROR] could not detect a RoCE v2 + IPv4 GID on one or both nodes."
    echo "        Check that both ports have an IPv4 address and that RoCE v2 is"
    echo "        enabled. Without a usable GID, multi-node RDMA cannot work."
    exit 1
fi

if [ "${IDX_A}" != "${IDX_B}" ]; then
    echo
    echo "[WARN] the two nodes report DIFFERENT indices (${IDX_A} vs ${IDX_B})."
    echo "       NCCL is given a single value, so one of them will be wrong."
    echo "       Align the node addressing / RoCE configuration before testing."
else
    echo
    echo "[OK] both nodes agree on index ${IDX_A}."
fi

if [ "${NO_TEST}" -eq 1 ]; then
    echo
    echo "Use it with:  NCCL_IB_GID_INDEX=${IDX_A}"
    exit 0
fi

# ---- confirm with a bandwidth test ----------------------------------------
# Resolve the peer netdev via ibdev2netdev / ndevs, not device/net: on a bonded
# RoCE setup device/net is the physical PF (no routable IP), not the bond.
PEER_IP="$(_ssh "${B}" "nd=\$(ibdev2netdev 2>/dev/null | awk '\$1==\"${DEV}\"{print \$5; exit}'); \
                        [ -n \"\$nd\" ] || nd=\$(cat /sys/class/infiniband/${DEV}/ports/1/gid_attrs/ndevs/${IDX_B:-0} 2>/dev/null); \
                        [ -n \"\$nd\" ] || nd=\$(ls /sys/class/infiniband/${DEV}/device/net 2>/dev/null | head -1); \
                        [ -n \"\$nd\" ] && ip -4 -o addr show \$nd 2>/dev/null | awk '{print \$4}' | cut -d/ -f1 | head -1")"
if [ -z "${PEER_IP}" ]; then
    echo "[ERROR] no IPv4 on ${B}:${DEV}; skipping the confirmation test" >&2
    exit 1
fi

echo
echo "--- confirmation: ib_write_bw with -x ${IDX_A} (peer IP ${PEER_IP}) ---"
_ssh "${B}" "pkill -x ib_write_bw 2>/dev/null; sleep 1; \
             (setsid nohup timeout 30 ib_write_bw -d ${DEV} -x ${IDX_A} -F --report_gbits >/tmp/gidchk.txt 2>&1 < /dev/null &)" >/dev/null 2>&1
sleep 3

SRV_LOG="$(_ssh "${B}" "head -20 /tmp/gidchk.txt 2>/dev/null")"
if echo "${SRV_LOG}" | grep -qiE "not found|Unable to find|command not found"; then
    echo "[FATAL] the peer server did not start; result invalid:" >&2
    echo "${SRV_LOG}" | sed 's/^/    /' >&2
    exit 1
fi

OUT="$(_ssh "${A}" "timeout 25 ib_write_bw -d ${DEV} -x ${IDX_A} -F --report_gbits ${PEER_IP} 2>&1 | tail -8")"
echo "${OUT}" | sed 's/^/  /'
_ssh "${B}" "pkill -x ib_write_bw 2>/dev/null; exit 0" >/dev/null 2>&1

echo
if echo "${OUT}" | grep -q "ETIMEDOUT\|Couldn't connect\|Unable to"; then
    echo "RESULT: still failing with index ${IDX_A}."
    echo "        The GID is not the problem. Look at: RoCE QoS settings (MTU, PFC"
    echo "        priority, trust mode) differing between nodes, VLAN/subnet"
    echo "        reachability, or rail isolation (see rail_isolation_check.sh)."
    exit 1
fi

echo "RESULT: RDMA works with GID index ${IDX_A}. Use:"
echo "        export NCCL_IB_GID_INDEX=${IDX_A}"
