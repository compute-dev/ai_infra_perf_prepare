#!/bin/bash
# ============================================================================
# perftest_pair.sh — per-rail point-to-point RDMA baseline between two nodes.
#
# For every RDMA device (rail), runs:
#     ib_write_bw              unidirectional Write bandwidth (Gb/s)
#     ib_write_bw -b           bidirectional Write bandwidth (Gb/s)
#     ib_read_bw               unidirectional Read bandwidth (Gb/s)
#     ib_write_lat             Write latency (us)
#     ib_send_lat              Send latency (us)
#
# and snapshots the RDMA hardware error counters on both nodes before the run
# and after each direction. A non-zero counter is the single best early
# indicator of a bad cable, port or fabric path, and it is cheap to collect.
#
# Why both directions: RDMA Write flows client -> server, so "forward" stresses
# node B's transmit side and node A's receive side, and "reverse" stresses the
# opposite. If one direction is bad, running the other plus comparing the two
# counter snapshots tells you which node and which side is at fault.
#
# Usage:
#     bash perftest_pair.sh <nodeA> <nodeB> [options]
#
# Options:
#     --mode forward|reverse|both   default both
#     --devices "d1 d2 ..."         RDMA devices to test; default = auto-discover
#                                   all devices with an ACTIVE port on nodeA
#     --gid N                       GID index (RoCE). Default: auto-detect per node
#     --size BYTES                  message size, default 8388608 (8 MB)
#     --iters N                     client iterations, default 20
#     --ssh-port N                  host ssh port, default 22
#     --ssh-user USER               default root
#     --results DIR                 output directory, default ./results/perftest
#
# Requirements on BOTH nodes: perftest installed, and memlock unlimited
# (`ulimit -l`), otherwise ibv_reg_mr fails. If perftest is not installed on the
# hosts, run this inside a container that has it, with --network=host
# --privileged --device=/dev/infiniband --ulimit memlock=-1.
#
# Output: $RESULTS/perftest_<ts>.log  raw output + counter snapshots
#         $RESULTS/summary_<ts>.txt  the per-rail table
# ============================================================================
set -uo pipefail

# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
if [ $# -lt 2 ]; then
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

NODE_A="$1"; shift
NODE_B="$1"; shift

MODE="both"
DEVLIST=""
GID=""
SZ=8388608
N=20
SSH_PORT="${SSH_PORT:-22}"
SSH_USER="${SSH_USER:-root}"
RESULTS="results/perftest"

while [ $# -gt 0 ]; do
    case "$1" in
        --mode)      MODE="${2:-}"; shift 2 ;;
        --devices)   DEVLIST="${2:-}"; shift 2 ;;
        --gid)       GID="${2:-}"; shift 2 ;;
        --size)      SZ="${2:-}"; shift 2 ;;
        --iters)     N="${2:-}"; shift 2 ;;
        --ssh-port)  SSH_PORT="${2:-}"; shift 2 ;;
        --ssh-user)  SSH_USER="${2:-}"; shift 2 ;;
        --results)   RESULTS="${2:-}"; shift 2 ;;
        -h|--help)   sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown option: $1 (use --help)" >&2; exit 1 ;;
    esac
done

case "${MODE}" in forward|reverse|both) ;; *) echo "[ERROR] --mode must be forward|reverse|both" >&2; exit 1 ;; esac

mkdir -p "${RESULTS}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="${RESULTS}/perftest_${TS}.log"
SUM="${RESULTS}/summary_${TS}.txt"

SSHX="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p ${SSH_PORT}"

# _ssh <node> <command>
_ssh() { ssh ${SSHX} "${SSH_USER}@$1" "$2" 2>&1; }

# _ssh_script <node> <local_script> [args...]
# Run a LOCAL script on a remote node, feeding it over stdin.
#
# The redirection must happen HERE. Writing `bash -s -- x < /local/path` as the
# remote command string makes the *remote* shell try to open a path that only
# exists on this machine — it fails with "No such file or directory".
_ssh_script() {
    local node="$1" script="$2"; shift 2
    ssh ${SSHX} "${SSH_USER}@${node}" "bash -s -- $*" < "${script}" 2>&1
}

# ---------------------------------------------------------------------------
# discover the device list and each device's rail IP on a node
#
# The RDMA path is pinned by each end's -d device; the IP is only the TCP
# control channel. So for a given rail we need the SERVER's IP on that rail.
# ---------------------------------------------------------------------------
_dev_ip() {   # _dev_ip <node> <dev>  -> IPv4 of the netdev carrying this rail
    # Prefer ibdev2netdev / the sysfs ndevs map over
    # /sys/class/infiniband/<dev>/device/net.
    #
    # On a bonded RoCE setup those disagree: device/net resolves to the physical
    # PF netdev (e.g. eth2) which carries no routable address, while the
    # RoCE-relevant netdev is the bond (e.g. bond2) that ibdev2netdev and
    # gid_attrs/ndevs report. Using device/net yields an empty IP for every rail.
    _ssh "$1" "nd=\$(ibdev2netdev 2>/dev/null | awk '\$1==\"$2\"{print \$5; exit}'); \
               [ -n \"\$nd\" ] || nd=\$(cat /sys/class/infiniband/$2/ports/1/gid_attrs/ndevs/${GID:-0} 2>/dev/null); \
               [ -n \"\$nd\" ] || nd=\$(ls /sys/class/infiniband/$2/device/net 2>/dev/null | head -1); \
               [ -n \"\$nd\" ] && ip -4 -o addr show \$nd 2>/dev/null | awk '{print \$4}' | cut -d/ -f1 | head -1"
}

discover_devices() {   # active-port devices on a node, sorted
    _ssh "$1" "for d in /sys/class/infiniband/*; do \
                 [ -e \"\$d/ports/1/state\" ] || continue; \
                 [ \"\$(cat \$d/ports/1/state 2>/dev/null)\" = '4: ACTIVE' ] && basename \$d; \
               done | sort -V"
}

echo "== perftest pair baseline  $(date)  mode=${MODE} ==" | tee "${LOG}"
echo "[INFO] nodes: A=${NODE_A} B=${NODE_B}  size=${SZ}B iters=${N} ssh_port=${SSH_PORT}" | tee -a "${LOG}"

# ---- device list -----------------------------------------------------------
if [ -z "${DEVLIST}" ]; then
    mapfile -t DEVLIST_ARR < <(discover_devices "${NODE_A}")
    if [ "${#DEVLIST_ARR[@]}" -eq 0 ]; then
        echo "[ERROR] no RDMA device with an ACTIVE port on ${NODE_A}." >&2
        echo "        Check: ls /sys/class/infiniband/  and  ibstat" >&2
        exit 1
    fi
    DEVLIST="${DEVLIST_ARR[*]}"
    echo "[INFO] auto-discovered ${#DEVLIST_ARR[@]} ACTIVE device(s) on ${NODE_A}: ${DEVLIST}" | tee -a "${LOG}"

    # The peer must expose the same rails, or only some pairs will be comparable
    PEER_DEVICES="$(discover_devices "${NODE_B}")"
    for d in ${DEVLIST}; do
        if ! grep -qx "${d}" <<<"${PEER_DEVICES}"; then
            echo "[WARN] ${d} is ACTIVE on ${NODE_A} but not on ${NODE_B} — this rail will fail" | tee -a "${LOG}"
        fi
    done
else
    echo "[INFO] using the supplied device list: ${DEVLIST}" | tee -a "${LOG}"
fi

# ---- GID -------------------------------------------------------------------
if [ -z "${GID}" ]; then
    LINK_LAYER="$(_ssh "${NODE_A}" "cat /sys/class/infiniband/$(echo "${DEVLIST}" | awk '{print $1}')/ports/1/link_layer 2>/dev/null")"
    if [ "${LINK_LAYER}" = "Ethernet" ]; then
        # RoCE: the GID index matters and must be right, or every QP fails
        GID_HELPER="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tools/detect_roce_gid.sh"
        GID_FIRST="$(echo "${DEVLIST}" | awk '{print $1}')"
        GID="$(_ssh_script "${NODE_A}" "${GID_HELPER}" "${GID_FIRST}" | tail -1)"
        if ! [[ "${GID}" =~ ^[0-9]+$ ]]; then
            echo "[ERROR] could not auto-detect the RoCE GID index (got: '${GID}')." >&2
            echo "        Run tools/detect_roce_gid.sh <dev> --table on a node, then pass --gid N." >&2
            exit 1
        fi
        echo "[INFO] RoCE fabric — auto-detected GID index ${GID}" | tee -a "${LOG}"
    else
        GID="0"
        echo "[INFO] ${LINK_LAYER} fabric — GID index not used (set to 0)" | tee -a "${LOG}"
    fi
fi

echo "[INFO] devices: ${DEVLIST}" | tee -a "${LOG}"
echo "[INFO] gid=${GID}  size=${SZ}  iters=${N}" | tee -a "${LOG}"

# ---- per-node dev -> rail IP maps ------------------------------------------
declare -A AIP BIP
for d in ${DEVLIST}; do
    AIP["${d}"]="$(_dev_ip "${NODE_A}" "${d}")"
    BIP["${d}"]="$(_dev_ip "${NODE_B}" "${d}")"
    echo "[INFO] ${d}: ${NODE_A}=${AIP[${d}]:-<no ip>}  ${NODE_B}=${BIP[${d}]:-<no ip>}" | tee -a "${LOG}"
done

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Snapshot the five RDMA hardware error counters on a node.
# All must stay zero. Any growth across a run points at that node's NIC/port.
snap_counters() {   # <node> <tag>
    local node=$1 tag=$2
    echo "---- hw_counters snapshot @ $node ($tag) $(date +%T) ----" >> "${LOG}"
    for d in ${DEVLIST}; do
        _ssh "${node}" "for f in out_of_sequence packet_seq_err req_cqe_error local_ack_timeout_err rnr_nak_retry_err; do \
                          v=\$(cat /sys/class/infiniband/${d}/ports/1/hw_counters/\$f 2>/dev/null); \
                          printf '%s=%s ' \"\$f\" \"\${v:-NA}\"; \
                        done; echo" \
            | sed "s/^/${d}: /" >> "${LOG}"
    done
}

# Kill leftover perftest processes on a node.
# MUST use `pkill -x` (exact process name). `pkill -f` also matches the
# `bash -c ...` command line of the ssh session itself, which kills our own
# shell and leaves the server never started.
kill_all() {
    _ssh "$1" "pkill -x ib_write_bw; pkill -x ib_read_bw; pkill -x ib_write_lat; pkill -x ib_send_lat; exit 0" >/dev/null 2>&1
}

# srv_up <dev> <extra args...> — start the server in the background on $SRV.
# It runs with a huge iteration count so it waits for the client rather than
# exiting between runs.
srv_up() {
    local dev=$1; shift
    _ssh "$SRV" "nohup ${BIN} $* -d ${dev} -x ${GID} -F >/dev/null 2>&1 & exit 0" >/dev/null 2>&1
}

# run_client <bin> <server_ip> <extra args...> — one client run, with a single
# retry on "Couldn't connect" to absorb the server's startup latency.
run_client() {
    local bin=$1 ip=$2; shift 2
    local out i
    for i in 1 2; do
        out="$(_ssh "$CLI" "timeout 60 ${bin} $* -d ${DEV} -x ${GID} -F ${ip}")"
        if ! grep -q "Couldn't connect" <<<"${out}"; then echo "${out}"; return 0; fi
        sleep 2
    done
    echo "${out}"; return 1
}

# Bandwidth row is " <size> <iters> <peak> <avg> <msgrate>": field 4 is avg Gb/s.
bw_avg() { awk -v sz="${SZ}" '$1==sz{print $4}' <<<"$1" | tail -1; }
# Latency row is "<size> <iters> <t_min> <t_max> <t_typ> <t_avg> <stdev>": field 6.
lat_avg() { awk '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {v=$6} END{print v}' <<<"$1"; }

# run_direction <server_node> <client_node> <server_map> <client_map> <label>
# Write data flows client -> server; Read data flows server -> client. The
# client always connects to the SERVER's rail IP, since the server's -d decides
# which rail it listens on.
run_direction() {
    SRV="$1"; CLI="$2"
    local -n SIPREF=$3
    local -n CIPREF=$4
    local DIRLABEL="$5"

    echo "" | tee -a "${SUM}"
    echo "== direction: ${DIRLABEL} (server=${SRV}, client=${CLI}, write flows ${CLI} -> ${SRV}) ==" | tee -a "${SUM}"
    printf "%-13s %-7s %11s %11s %11s %11s %11s\n" "rail(dev)" "GB/s" "wr_bw Gb/s" "wr_bidi" "rd_bw" "wr_lat us" "send_lat" | tee -a "${SUM}"
    printf "%-13s %-7s %11s %11s %11s %11s %11s\n" "-----------" "-----" "----------" "--------" "------" "--------" "-------" >> "${SUM}"

    for DEV in ${DEVLIST}; do
        S="${SIPREF[${DEV}]:-}"
        if [ -z "${S}" ]; then
            echo "[WARN] no rail IP for ${DEV} on ${SRV} — skipping" | tee -a "${SUM}"
            continue
        fi
        echo "==== [${DIRLABEL}] ${DEV}  srv=${S}(${SRV}) cli=${CIPREF[${DEV}]:-?}(${CLI}) ====" >> "${LOG}"

        # --- write bandwidth, unidirectional ---
        BIN=ib_write_bw
        kill_all "${SRV}"; sleep 0.5
        srv_up "${DEV}" --report_gbits -s "${SZ}" -n 100000; sleep 1.5
        OUT="$(run_client ib_write_bw "${S}" --report_gbits -s "${SZ}" -n "${N}")"
        WB="$(bw_avg "${OUT}")"; [ -z "${WB}" ] && WB=FAIL
        echo "--- write_bw ---"$'\n'"${OUT}" >> "${LOG}"
        kill_all "${SRV}"

        # --- write bandwidth, bidirectional ---
        srv_up "${DEV}" --report_gbits -s "${SZ}" -n 100000 -b; sleep 1.5
        OUT="$(run_client ib_write_bw "${S}" --report_gbits -s "${SZ}" -n "${N}" -b)"
        WBB="$(bw_avg "${OUT}")"; [ -z "${WBB}" ] && WBB=FAIL
        echo "--- write_bw -b (bidir) ---"$'\n'"${OUT}" >> "${LOG}"
        kill_all "${SRV}"

        # --- read bandwidth ---
        BIN=ib_read_bw
        srv_up "${DEV}" --report_gbits -s "${SZ}" -n 100000; sleep 1.5
        OUT="$(run_client ib_read_bw "${S}" --report_gbits -s "${SZ}" -n "${N}")"
        RB="$(bw_avg "${OUT}")"; [ -z "${RB}" ] && RB=FAIL
        echo "--- read_bw ---"$'\n'"${OUT}" >> "${LOG}"
        kill_all "${SRV}"

        # --- write latency ---
        BIN=ib_write_lat
        srv_up "${DEV}"; sleep 1.5
        OUT="$(run_client ib_write_lat "${S}")"
        WL="$(lat_avg "${OUT}")"; [ -z "${WL}" ] && WL=FAIL
        echo "--- write_lat ---"$'\n'"${OUT}" >> "${LOG}"
        kill_all "${SRV}"

        # --- send latency ---
        BIN=ib_send_lat
        srv_up "${DEV}"; sleep 1.5
        OUT="$(run_client ib_send_lat "${S}")"
        SL="$(lat_avg "${OUT}")"; [ -z "${SL}" ] && SL=FAIL
        echo "--- send_lat ---"$'\n'"${OUT}" >> "${LOG}"
        kill_all "${SRV}"

        if [ "${WB}" != FAIL ]; then WBG="$(awk -v v="${WB}" 'BEGIN{printf "%.1f", v/8}')"; else WBG=FAIL; fi
        printf "%-13s %-7s %11s %11s %11s %11s %11s\n" "${DEV}" "${WBG}" "${WB}" "${WBB}" "${RB}" "${WL}" "${SL}" | tee -a "${SUM}"
    done

    # Counter snapshot after this direction — compare against "before"
    snap_counters "${NODE_A}" "after-${DIRLABEL}"
    snap_counters "${NODE_B}" "after-${DIRLABEL}"
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
PERFTEST_VER="$(_ssh "${NODE_B}" "ib_write_bw --version 2>/dev/null | tail -1")"
echo "[INFO] perftest on ${NODE_B}: ${PERFTEST_VER:-<not found>}" | tee -a "${LOG}"
if [ -z "${PERFTEST_VER}" ]; then
    echo "[ERROR] ib_write_bw not found on ${NODE_B}." >&2
    echo "        Install perftest on both nodes, or run this inside a container that has it." >&2
    exit 1
fi

# Baseline snapshot first, so pre-existing errors are visible too
snap_counters "${NODE_A}" "before"
snap_counters "${NODE_B}" "before"

case "${MODE}" in
    forward)
        run_direction "${NODE_A}" "${NODE_B}" AIP BIP "forward: server=${NODE_A}, client=${NODE_B}" ;;
    reverse)
        run_direction "${NODE_B}" "${NODE_A}" BIP AIP "reverse: server=${NODE_B}, client=${NODE_A}" ;;
    both)
        run_direction "${NODE_A}" "${NODE_B}" AIP BIP "forward: server=${NODE_A}, client=${NODE_B}"
        run_direction "${NODE_B}" "${NODE_A}" BIP AIP "reverse: server=${NODE_B}, client=${NODE_A}" ;;
esac

kill_all "${NODE_A}"; kill_all "${NODE_B}"

echo "" | tee -a "${SUM}"
echo "done. mode=${MODE} raw=${LOG} summary=${SUM}" | tee -a "${SUM}"
echo "" | tee -a "${SUM}"
echo "Check the raw log for 'hw_counters snapshot' blocks. All five counters" | tee -a "${SUM}"
echo "(out_of_sequence, packet_seq_err, req_cqe_error, local_ack_timeout_err," | tee -a "${SUM}"
echo "rnr_nak_retry_err) must be 0 on every rail, before and after." | tee -a "${SUM}"
echo "Compare the measured Gb/s against your link rate — a 200G port should" | tee -a "${SUM}"
echo "read ~193 Gb/s per single QP. See readme.md for baselines." | tee -a "${SUM}"
