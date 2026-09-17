#!/bin/bash
# ============================================================================
# cluster_runtime_check.sh
#   Cluster "software / runtime-state" consistency check -- the cluster version
#   of collect_runtime_state.sh. Complements cluster_spec_check.sh (hardware
#   spec); together they form a complete server-config report.
#
#   Flow:
#     1) Pipe collect_runtime_state.sh -j to each node over plain ssh (stdin),
#        collecting one single-line JSON per node.
#     2) For each field take the mode (most frequent value) as the baseline and
#        flag the nodes that differ.
#     3) A difference on a critical field (driver/libcuda/ECC/HCA/OFED/clock/power/...)
#        => FAIL (exit 2). Advisory fields (os/persistence/memlock/image/ecc-cumulative)
#        only warn.
#     4) Emit a consistency report + optional CSV.
#
#   Transport: plain ssh only. The collector script is fed to each node's bash
#   over stdin, so no shared filesystem and no parallel-ssh fan-out tool are
#   required. The hosts file is required (-H) and lists one SSH-reachable
#   address per line.
#
# Usage:
#     bash cluster_runtime_check.sh -H hosts.txt                         # all nodes, port 22, root
#     bash cluster_runtime_check.sh -H hosts.txt -P 2222                 # custom SSH port
#     bash cluster_runtime_check.sh -H hosts.txt -u admin               # non-root user
#     bash cluster_runtime_check.sh -H hosts.txt -o rt.csv              # also write CSV
#
# Exit codes: 0 = all critical fields agree; 2 = a critical field differs or a
#             node is unreachable; 1 = arg/dependency error.
# ============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COLLECT="${SCRIPT_DIR}/collect_runtime_state.sh"

HOSTS_FILE=""
SSH_PORT="22"
SSH_USER="root"
CSV_OUT=""
while getopts "H:P:u:o:h" opt; do
    case $opt in
        H) HOSTS_FILE="$OPTARG" ;;
        P) SSH_PORT="$OPTARG" ;;
        u) SSH_USER="$OPTARG" ;;
        o) CSV_OUT="$OPTARG" ;;
        h) sed -n '2,31p' "$0"; exit 0 ;;
    esac
done

if [ -t 1 ]; then
    C_RED='\033[0;31m'; C_GRN='\033[0;32m'; C_YLW='\033[0;33m'
    C_BLU='\033[0;34m'; C_CYN='\033[0;36m'; C_BLD='\033[1m'; C_RST='\033[0m'
else
    C_RED=''; C_GRN=''; C_YLW=''; C_BLU=''; C_CYN=''; C_BLD=''; C_RST=''
fi
log_info() { echo -e "${C_CYN}[INFO]${C_RST}  $*"; }
log_warn() { echo -e "${C_YLW}[WARN]${C_RST}  $*"; }
log_err()  { echo -e "${C_RED}[ERROR]${C_RST} $*" >&2; }
log_head() { echo -e "\n${C_BLD}${C_BLU}==== $* ====${C_RST}"; }
log_ok()   { echo -e "${C_GRN}[ OK ]${C_RST}  $*"; }
log_diff() { echo -e "${C_RED}[DIFF]${C_RST}  $*"; }

# -H is required
if [ -z "$HOSTS_FILE" ]; then
    log_err "Usage: $0 -H <hosts_file> [-P <port>] [-u <user>] [-o <csv>]"
    log_err "  -H <hosts_file>  (required) one SSH-reachable address per line; '#' comments allowed"
    log_err "  -P <port>        SSH port (default 22)"
    log_err "  -u <user>        SSH user (default root)"
    log_err "  -o <csv>         also write a CSV report to this path"
    exit 1
fi

[ -f "$COLLECT" ]   || { log_err "collector script not found: $COLLECT"; exit 1; }
[ -f "$HOSTS_FILE" ] || { log_err "hosts file not found: $HOSTS_FILE"; exit 1; }

mapfile -t NODES < <(grep -vE '^\s*(#|$)' "$HOSTS_FILE" | awk '{print $1}')
NODE_CNT=${#NODES[@]}
[ "$NODE_CNT" -gt 0 ] || { log_err "no valid nodes in hosts file"; exit 1; }

SSHX="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p${SSH_PORT}"

log_head "Cluster runtime-state consistency check"
log_info "nodes      : ${NODE_CNT}"
log_info "HOSTS       : ${HOSTS_FILE}"
log_info "SSH port    : ${SSH_PORT}"
log_info "SSH user    : ${SSH_USER}"
log_info "collector   : ${COLLECT} (piped to each node over stdin)"

# ---------------------------- Step 1. concurrent collection ----------------------------
log_head "Step 1. Collect JSON from each node"
WORK=$(mktemp -d -t cluster_rt_XXXXXX)
trap 'rm -rf "$WORK"' EXIT
OUTDIR="$WORK/out"; mkdir -p "$OUTDIR"

log_info "transport: plain ssh (no shared fs; script piped over stdin)"
pids=()
for h in "${NODES[@]}"; do
    (
        timeout 180 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
            -p "${SSH_PORT}" "${SSH_USER}@${h}" 'bash -s -- -j' \
            < "$COLLECT" > "$OUTDIR/$h" 2>"$OUTDIR/$h.err"
    ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done

# ---------------------------- Step 2. parse ----------------------------
log_head "Step 2. Parse each node JSON"

# field -> label -> critical (1=critical, a diff => FAIL; 0=advisory, warn only)
FIELDS=(driver cuda_api libcuda cuda_compat gpu_count ecc_mode sm_clock_max power_limit \
        fabric_type compute_hca compute_active compute_rate compute_mtu compute_pcie compute_fw \
        roce_gid_v2idx roce_trust roce_pfc_prio ofed fabricmanager fm_active peermem kernel \
        gpu_nic_rail gpu_nic_topo lldp_switch nvlink_links nvlink_speed hca_devices os persistence memlock numa_nodes ecc_uncorr bench_image_id)
declare -A LABEL=(
  [driver]="driver_version" [cuda_api]="CUDA API max" [libcuda]="userspace libcuda" [cuda_compat]="cuda-compat"
  [gpu_count]="GPU count" [ecc_mode]="ECC mode" [sm_clock_max]="SM clock max" [power_limit]="power limit"
  [fabric_type]="Fabric type (IB/RoCE)" [compute_hca]="Compute HCA list" [compute_active]="Compute HCA ACTIVE"
  [compute_rate]="Compute HCA rate" [compute_mtu]="Compute HCA MTU" [compute_pcie]="Compute HCA PCIe" [compute_fw]="Compute HCA firmware"
  [roce_gid_v2idx]="RoCE v2 GID index" [roce_trust]="RoCE trust mode" [roce_pfc_prio]="RoCE PFC priority"
  [ofed]="OFED" [fabricmanager]="FabricManager" [fm_active]="FM running state" [peermem]="peermem" [kernel]="kernel"
  [gpu_nic_rail]="GPU-NIC rail topology" [gpu_nic_topo]="GPU-NIC full signature" [lldp_switch]="LLDP switch neighbour"
  [nvlink_links]="NVLink active count" [nvlink_speed]="NVLink per-link speed" [hca_devices]="All HCA list (incl. storage)"
  [os]="OS" [persistence]="persistence" [memlock]="memlock" [numa_nodes]="NUMA nodes"
  [ecc_uncorr]="ECC cumulative uncorrected" [bench_image_id]="bench image id" )
declare -A CRIT=(
  [driver]=1 [cuda_api]=1 [libcuda]=1 [cuda_compat]=1 [gpu_count]=1 [ecc_mode]=1 [sm_clock_max]=1 [power_limit]=1
  [fabric_type]=1 [compute_hca]=1 [compute_active]=1 [compute_rate]=1 [compute_mtu]=1 [compute_pcie]=1 [compute_fw]=1
  [roce_gid_v2idx]=1 [roce_trust]=1 [roce_pfc_prio]=1 [ofed]=1 [fabricmanager]=1 [fm_active]=1 [peermem]=1 [kernel]=1
  [gpu_nic_rail]=1 [gpu_nic_topo]=0 [lldp_switch]=0 [nvlink_links]=0 [nvlink_speed]=0 [hca_devices]=0 [os]=0 [persistence]=0 [memlock]=0 [numa_nodes]=0 [ecc_uncorr]=0 [bench_image_id]=0 )

declare -A VAL REACHABLE
for h in "${NODES[@]}"; do
    f="$OUTDIR/$h"
    line=""; [ -s "$f" ] && line=$(grep -E '^\{' "$f" | tail -n1)
    if [ -z "$line" ]; then REACHABLE["$h"]=0; log_warn "node ${h}: no JSON (unreachable / collection failed)"; continue; fi
    REACHABLE["$h"]=1
    for fld in "${FIELDS[@]}"; do
        VAL["$h|$fld"]=$(echo "$line" | sed -nE "s/.*\"${fld}\":\"([^\"]*)\".*/\1/p")
    done
done

# ---------------------------- Step 3. per-field consistency ----------------------------
log_head "Step 3. Consistency results (star=critical field; a diff => FAIL)"
FAIL=0
declare -A BASELINE
for fld in "${FIELDS[@]}"; do
    label="${LABEL[$fld]}"; crit="${CRIT[$fld]}"
    mapfile -t vals < <(for h in "${NODES[@]}"; do [ "${REACHABLE[$h]:-0}" -eq 1 ] && printf '%s\n' "${VAL["$h|$fld"]}"; done)
    [ "${#vals[@]}" -eq 0 ] && { BASELINE[$fld]="<no data>"; continue; }
    base=$(printf '%s\n' "${vals[@]}" | sort | uniq -c | sort -rn | head -n1 | sed -E 's/^[[:space:]]*[0-9]+[[:space:]]+//')
    BASELINE[$fld]="$base"
    diff_hosts=()
    for h in "${NODES[@]}"; do
        [ "${REACHABLE[$h]:-0}" -eq 1 ] || continue
        [ "${VAL["$h|$fld"]}" != "$base" ] && diff_hosts+=("$h")
    done
    mark=""; [ "$crit" -eq 1 ] && mark="*"
    if [ "${#diff_hosts[@]}" -eq 0 ]; then
        # ecc_uncorr all-agree but non-zero still deserves a warning
        if [ "$fld" = "ecc_uncorr" ] && [ "$base" != "0" ] && [ "$base" != "N/A" ]; then
            log_warn "${mark}${label}  all agree but non-zero (baseline ${base}) -- watch for ECC degradation"
        else
            log_ok "${mark}${label}  all agree  ->  ${base}"
        fi
    else
        if [ "$crit" -eq 1 ]; then FAIL=1; log_diff "${mark}${label}  diff present"; else log_warn "${label}  diff present (advisory, no FAIL)"; fi
        echo -e "       baseline: ${C_GRN}${base}${C_RST}"
        for h in "${diff_hosts[@]}"; do
            echo -e "       ${C_RED}x ${h}${C_RST} : ${VAL["$h|$fld"]}"
        done
    fi
done

# Unreachable nodes
UNREACH=(); for h in "${NODES[@]}"; do [ "${REACHABLE[$h]:-0}" -eq 0 ] && UNREACH+=("$h"); done
if [ "${#UNREACH[@]}" -gt 0 ]; then
    FAIL=1; log_diff "unreachable nodes (${#UNREACH[@]}/${NODE_CNT}):"
    for h in "${UNREACH[@]}"; do echo -e "       ${C_RED}x ${h}${C_RST}"; done
fi

# ---------------------------- Step 4. CSV (optional) ----------------------------
if [ -n "$CSV_OUT" ]; then
    {
        printf 'host,reachable'; for fld in "${FIELDS[@]}"; do printf ',%s' "$fld"; done; printf '\n'
        for h in "${NODES[@]}"; do
            printf '%s,%s' "$h" "${REACHABLE[$h]:-0}"
            for fld in "${FIELDS[@]}"; do
                v="${VAL["$h|$fld"]:-}"
                if [[ "$v" == *,* || "$v" == *\"* ]]; then v=${v//\"/\"\"}; printf ',"%s"' "$v"; else printf ',%s' "$v"; fi
            done
            printf '\n'
        done
    } > "$CSV_OUT"
    log_info "CSV written to: ${CSV_OUT}"
fi

# ---------------------------- Step 5. summary ----------------------------
log_head "Summary"
REACH_CNT=0; for h in "${NODES[@]}"; do [ "${REACHABLE[$h]:-0}" -eq 1 ] && REACH_CNT=$((REACH_CNT+1)); done
log_info "covered ${REACH_CNT}/${NODE_CNT} nodes"
if [ "$FAIL" -eq 0 ]; then
    echo -e "${C_GRN}${C_BLD}OK: all critical runtime-state fields agree; safe to proceed to cuBLAS/training tests${C_RST}"
    exit 0
else
    echo -e "${C_RED}${C_BLD}FAIL: a critical field differs or a node is unreachable -- fix/exclude before testing (see the * items above)${C_RST}"
    exit 2
fi
