#!/bin/bash
# ============================================================================
# perftest_agg.sh — per-rail aggregate RDMA bandwidth (multi-QP, one process).
#
# perftest_pair.sh measures a single QP, which is how you find a dead port. This
# script measures the AGGREGATE capacity of a rail, which is what NCCL actually
# uses.
#
# Method: one `ib_write_bw -q <QPS>` process per rail. Several QPs in one
# process are hashed across the bond's physical ports by the NIC, so this
# exercises the whole rail rather than one member port.
#
# !! Judge on PEAK, and run at least 3 times, taking the maximum. !!
# On a bonded (802.3ad / mlx5 LAG) rail the hash is fixed for a given QP set, so
# a single run can legitimately put every QP on one physical port and report
# roughly half the true capacity. That is hash luck, not a fault — it is exactly
# why the peak over several runs is the right statistic.
#
# Note: do NOT sum the peaks of N independent short single-QP streams to get
# "aggregate". That over-reports badly (it can exceed the physical ceiling of
# the rail) because the short runs never reach a coordinated steady state.
# Use this script instead.
#
# Usage:
#     bash perftest_agg.sh <nodeA> <nodeB> [options]
#
# Options:
#     --devices "d1 d2 ..."   RDMA devices to test; default auto-discover on nodeA
#     --gid N                 GID index (RoCE); default auto-detect
#     --size BYTES            message size, default 8388608 (8 MB)
#     --qps N                 QPs per rail, default 4
#     --runs N                repetitions per rail, default 3 (max taken)
#     --ssh-port N            host ssh port, default 22
#     --ssh-user USER         default root
#     --results DIR           output directory, default ./results/perftest
#
# Output: $RESULTS/summary_agg_<ts>.txt
#
# Requirements on both nodes: perftest, and memlock unlimited (`ulimit -l`).
# ============================================================================
set -uo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

NODE_A="$1"; shift
NODE_B="$1"; shift

DEVLIST=""
GID=""
SZ=8388608
QPS=4
RUNS=3
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-root}"
RESULTS="results/perftest"

while [ $# -gt 0 ]; do
    case "$1" in
        --devices)   DEVLIST="${2:-}"; shift 2 ;;
        --gid)       GID="${2:-}"; shift 2 ;;
        --size)      SZ="${2:-}"; shift 2 ;;
        --qps)       QPS="${2:-}"; shift 2 ;;
        --runs)      RUNS="${2:-}"; shift 2 ;;
        --ssh-port)  SSH_PORT="${2:-}"; shift 2 ;;
        --ssh-user)  SSH_USER="${2:-}"; shift 2 ;;
        --results)   RESULTS="${2:-}"; shift 2 ;;
        -h|--help)   sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown option: $1 (use --help)" >&2; exit 1 ;;
    esac
done

mkdir -p "${RESULTS}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${RESULTS}/perftest_agg_${TS}.log"
SUM="${RESULTS}/summary_agg_${TS}.txt"

SSHX="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p ${SSH_PORT}"
_ssh() { ssh ${SSHX} "${SSH_USER}@$1" "$2" 2>&1; }

# _ssh_script <node> <local_script> [args...] — run a LOCAL script on a remote
# node via stdin. The redirection must be local; inside the remote command
# string the remote shell would look for a path that only exists here.
_ssh_script() {
    local node="$1" script="$2"; shift 2
    ssh ${SSHX} "${SSH_USER}@${node}" "bash -s -- $*" < "${script}" 2>&1
}

echo "== perftest aggregate (multi-QP)  $(date) ==" | tee "${LOG}"
echo "[INFO] nodes: A=${NODE_A} B=${NODE_B}  size=${SZ}B qps=${QPS} runs=${RUNS} ssh_port=${SSH_PORT}" | tee -a "${LOG}"

# ---- devices ---------------------------------------------------------------
if [ -z "${DEVLIST}" ]; then
    mapfile -t DEVLIST_ARR < <(_ssh "${NODE_A}" "for d in /sys/class/infiniband/*; do \
        [ -e \"\$d/ports/1/state\" ] || continue; \
        [ \"\$(cat \$d/ports/1/state 2>/dev/null)\" = '4: ACTIVE' ] && basename \$d; \
      done | sort -V")
    if [ "${#DEVLIST_ARR[@]}" -eq 0 ]; then
        echo "[ERROR] no RDMA device with an ACTIVE port on ${NODE_A}" >&2
        exit 1
    fi
    DEVLIST="${DEVLIST_ARR[*]}"
    echo "[INFO] auto-discovered devices: ${DEVLIST}" | tee -a "${LOG}"
fi

# ---- GID -------------------------------------------------------------------
if [ -z "${GID}" ]; then
    GID_FIRST="$(echo "${DEVLIST}" | awk '{print $1}')"
    LINK_LAYER="$(_ssh "${NODE_A}" "cat /sys/class/infiniband/${GID_FIRST}/ports/1/link_layer 2>/dev/null")"
    if [ "${LINK_LAYER}" = "Ethernet" ]; then
        GID_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tools/detect_roce_gid.sh"
        GID="$(_ssh_script "${NODE_A}" "${GID_HELPER}" "${GID_FIRST}" | tail -1)"
        if ! [[ "${GID}" =~ ^[0-9]+$ ]]; then
            echo "[ERROR] could not auto-detect the RoCE GID index (got '${GID}'); pass --gid N" >&2
            exit 1
        fi
        echo "[INFO] RoCE fabric — GID index ${GID}" | tee -a "${LOG}"
    else
        GID="0"
        echo "[INFO] ${LINK_LAYER} fabric — GID index not used" | tee -a "${LOG}"
    fi
fi

# The server's rail IP is what the client connects to for the control channel.
# Resolve the netdev via ibdev2netdev / ndevs, NOT device/net: on a bonded RoCE
# setup device/net gives the physical PF (no routable IP) instead of the bond.
declare -A SIP
for d in ${DEVLIST}; do
    SIP["${d}"]="$(_ssh "${NODE_B}" "nd=\$(ibdev2netdev 2>/dev/null | awk '\$1==\"${d}\"{print \$5; exit}'); \
                                     [ -n \"\$nd\" ] || nd=\$(cat /sys/class/infiniband/${d}/ports/1/gid_attrs/ndevs/${GID:-0} 2>/dev/null); \
                                     [ -n \"\$nd\" ] || nd=\$(ls /sys/class/infiniband/${d}/device/net 2>/dev/null | head -1); \
                                     [ -n \"\$nd\" ] && ip -4 -o addr show \$nd 2>/dev/null | awk '{print \$4}' | cut -d/ -f1 | head -1")"
done

kill_all() {
    _ssh "$1" "pkill -x ib_write_bw; exit 0" >/dev/null 2>&1
}

# ---------------------------------------------------------------------------
echo "" | tee -a "${SUM}"
printf "%-13s %12s %12s %12s %10s\n" "rail(dev)" "peak Gb/s" "avg Gb/s" "runs" "GB/s" | tee -a "${SUM}"
printf "%-13s %12s %12s %12s %10s\n" "-----------" "---------" "--------" "----" "----" >> "${SUM}"

TOTAL_PEAK=0
for DEV in ${DEVLIST}; do
    S="${SIP[${DEV}]:-}"
    if [ -z "${S}" ]; then
        echo "[WARN] no rail IP for ${DEV} on ${NODE_B} — skipping" | tee -a "${SUM}"
        continue
    fi

    BEST_PEAK=0
    BEST_AVG=0
    OK_RUNS=0

    for r in $(seq 1 "${RUNS}"); do
        kill_all "${NODE_B}"; sleep 0.5

        # Server: long-lived so it waits for each client run
        _ssh "${NODE_B}" "nohup ib_write_bw -d ${DEV} -x ${GID} -F --report_gbits -s ${SZ} -n 1000000 -q ${QPS} >/dev/null 2>&1 & exit 0" >/dev/null 2>&1
        sleep 1.5

        OUT="$(_ssh "${NODE_A}" "timeout 90 ib_write_bw -d ${DEV} -x ${GID} -F --report_gbits -s ${SZ} -n 200 -q ${QPS} ${S}")"
        echo "--- ${DEV} run ${r}/${RUNS} ---"$'\n'"${OUT}" >> "${LOG}"

        # Row: " <size> <iters> <peak> <avg> <msgrate>" -> peak=$3 avg=$4
        read -r PEAK AVG <<<"$(awk -v sz="${SZ}" '$1==sz{print $3, $4}' <<<"${OUT}" | tail -1)"

        if [ -n "${PEAK:-}" ]; then
            OK_RUNS=$((OK_RUNS + 1))
            if awk -v a="${PEAK}" -v b="${BEST_PEAK}" 'BEGIN{exit !(a>b)}'; then BEST_PEAK="${PEAK}"; BEST_AVG="${AVG}"; fi
            printf "    run %d: peak=%s avg=%s Gb/s\n" "${r}" "${PEAK}" "${AVG}" >> "${LOG}"
        else
            printf "    run %d: FAILED\n" "${r}" >> "${LOG}"
        fi
    done

    kill_all "${NODE_B}"

    if [ "${OK_RUNS}" -eq 0 ]; then
        printf "%-13s %12s %12s %12s %10s\n" "${DEV}" "FAIL" "-" "0/${RUNS}" "-" | tee -a "${SUM}"
        continue
    fi

    GBPS="$(awk -v v="${BEST_PEAK}" 'BEGIN{printf "%.1f", v/8}')"
    printf "%-13s %12.2f %12.2f %12s %10s\n" "${DEV}" "${BEST_PEAK}" "${BEST_AVG}" "${OK_RUNS}/${RUNS}" "${GBPS}" | tee -a "${SUM}"
    TOTAL_PEAK="$(awk -v a="${TOTAL_PEAK}" -v b="${BEST_PEAK}" 'BEGIN{printf "%.2f", a+b}')"

    if [ "${OK_RUNS}" -lt "${RUNS}" ]; then
        echo "    [NOTE] ${OK_RUNS}/${RUNS} runs produced a number for ${DEV}" >> "${SUM}"
    fi
done

echo "" | tee -a "${SUM}"
TOTAL_GB="$(awk -v v="${TOTAL_PEAK}" 'BEGIN{printf "%.1f", v/8}')"
printf "%-13s %12.2f %12s %12s %10s\n" "TOTAL" "${TOTAL_PEAK}" "" "" "${TOTAL_GB}" | tee -a "${SUM}"
echo "" | tee -a "${SUM}"
echo "Judgement:" | tee -a "${SUM}"
echo "  * Compare each rail's PEAK against the rail's physical rate. On a bonded" | tee -a "${SUM}"
echo "    802.3ad rail of 2x200G, expect ~390 Gb/s peak, NOT ~195 (which means" | tee -a "${SUM}"
echo "    all QPs hashed onto one member port — re-run before calling it a fault)." | tee -a "${SUM}"
echo "  * A rail whose peak stays at half the expected value across ${RUNS} runs is a" | tee -a "${SUM}"
echo "    real single-port problem: check that member port's link state and cable." | tee -a "${SUM}"
echo "  * Cross-check with perftest_pair.sh (single QP) and with the RDMA error" | tee -a "${SUM}"
echo "    counters, which must remain zero." | tee -a "${SUM}"
