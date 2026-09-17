#!/bin/bash
# ============================================================================
# split_ip.sh — generate power-of-two host lists from a full node list.
#
# NCCL tests are usually run at 1, 2, 4, 8, ... nodes so the bandwidth curve
# can be read as a function of scale. This reads your hosts file and writes
# host<GPU_count>.txt files, one per power-of-two step, plus host_all.txt.
#
# Split strategy, chosen by GPU_TYPE in set_env.sh:
#
#   1) Standard nodes (any GPU_TYPE not starting with "GB"):
#      take the first N IPs in file order.
#
#   2) GB200 / GB300 racks (GPU_TYPE starts with "GB"):
#      a rack holds 18 trays but only the first 16 are used, so the last 2 are
#      skipped. That keeps every step a power of two AND balanced across racks.
#      Example, with hosts listed rack by rack (18 consecutive IPs per rack):
#        host64.txt  -> first 16 IPs of rack 0
#        host128.txt -> first 16 IPs of rack 0 and of rack 1
#        host256.txt -> first 16 IPs of racks 0..3
#
# Also writes host_all.txt with every IP.
#
# Usage:
#   bash split_ip.sh [hosts_file] [output_dir]
#     hosts_file  default ./hosts.txt
#     output_dir  default the script directory
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------- load set_env.sh ----------------
ENV_FILE="${SCRIPT_DIR}/set_env.sh"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] ${ENV_FILE} not found — copy set_env.example.sh to set_env.sh and edit it"
    exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

# ---------------- validate GPUS_PER_NODE ----------------
if [[ -z "${GPUS_PER_NODE:-}" ]]; then
    echo "[ERROR] GPUS_PER_NODE is not set in set_env.sh"
    exit 1
fi
if ! [[ "${GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${GPUS_PER_NODE}" -le 0 ]]; then
    echo "[ERROR] invalid GPUS_PER_NODE: ${GPUS_PER_NODE}"
    exit 1
fi

# ---------------- validate GPU_TYPE ----------------
GPU_TYPE="${GPU_TYPE:-}"
if [[ -z "${GPU_TYPE}" ]]; then
    echo "[ERROR] GPU_TYPE is not set in set_env.sh (e.g. H200 | B200 | B300 | GB200 | GB300)"
    exit 1
fi

GPU_TYPE_UPPER="$(echo "${GPU_TYPE}" | tr '[:lower:]' '[:upper:]')"

IS_GB=0
TRAYS_PER_RACK="${TRAYS_PER_RACK:-18}"
USED_TRAYS_PER_RACK="${USED_TRAYS_PER_RACK:-16}"
if [[ "${GPU_TYPE_UPPER}" == GB* ]]; then
    IS_GB=1
    # GB200 / GB300 share the rack layout: 18 trays per rack, only 16 used
    if [[ "${GPUS_PER_NODE}" -ne 4 ]]; then
        echo "[WARN] GPU_TYPE=${GPU_TYPE} normally implies GPUS_PER_NODE=4; found ${GPUS_PER_NODE} — please double-check set_env.sh"
    fi
fi

# ---------------- arguments ----------------
HOSTS_FILE="${1:-${SCRIPT_DIR}/hosts.txt}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}}"

if [[ ! -f "${HOSTS_FILE}" ]]; then
    echo "[ERROR] hosts file not found: ${HOSTS_FILE}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ---------------- read and clean the node list ----------------
mapfile -t IPS < <(grep -vE '^[[:space:]]*(#|$)' "${HOSTS_FILE}" | awk '{print $1}')

TOTAL_IPS=${#IPS[@]}
if [[ "${TOTAL_IPS}" -eq 0 ]]; then
    echo "[ERROR] no usable entries in ${HOSTS_FILE}"
    exit 1
fi

echo "[INFO] ${HOSTS_FILE}: ${TOTAL_IPS} nodes"
echo "[INFO] GPU_TYPE      = ${GPU_TYPE}"
echo "[INFO] GPUS_PER_NODE = ${GPUS_PER_NODE}"
if [[ "${IS_GB}" -eq 1 ]]; then
    echo "[INFO] split strategy = GB rack mode (${TRAYS_PER_RACK} trays/rack, first ${USED_TRAYS_PER_RACK} used, last $((TRAYS_PER_RACK - USED_TRAYS_PER_RACK)) skipped)"
else
    echo "[INFO] split strategy = sequential"
fi
echo "[INFO] output dir     = ${OUTPUT_DIR}"
echo "----------------------------------------------------------------"

# ---------------- host_all.txt ----------------
ALL_FILE="${OUTPUT_DIR}/host_all.txt"
printf "%s\n" "${IPS[@]}" > "${ALL_FILE}"
echo "[OK] ${ALL_FILE} (${TOTAL_IPS} nodes, $((TOTAL_IPS * GPUS_PER_NODE)) GPUs)"

# ---------------- picker ----------------
# pick_ips <node_count>
#   standard mode: IPS[0 .. node_count-1]
#   GB mode      : walk rack by rack, taking at most USED_TRAYS_PER_RACK per
#                  rack and skipping the unused trailing trays
#   Output goes to the global PICKED array. Returns 1 if there are not enough IPs.
pick_ips() {
    local need=$1
    PICKED=()

    if [[ "${IS_GB}" -eq 0 ]]; then
        if [[ "${need}" -gt "${TOTAL_IPS}" ]]; then
            return 1
        fi
        PICKED=("${IPS[@]:0:${need}}")
        return 0
    fi

    local remain=${need}
    local rack_idx=0
    while [[ "${remain}" -gt 0 ]]; do
        local rack_start=$(( rack_idx * TRAYS_PER_RACK ))
        local take=${USED_TRAYS_PER_RACK}
        if [[ "${remain}" -lt "${take}" ]]; then
            take=${remain}
        fi
        local rack_end=$(( rack_start + take ))
        if [[ "${rack_end}" -gt "${TOTAL_IPS}" ]]; then
            return 1
        fi
        PICKED+=("${IPS[@]:${rack_start}:${take}}")
        remain=$(( remain - take ))
        rack_idx=$(( rack_idx + 1 ))
    done
    return 0
}

# ---------------- emit power-of-two steps ----------------
node_count=1
while : ; do
    gpu_count=$(( node_count * GPUS_PER_NODE ))
    out_file="${OUTPUT_DIR}/host${gpu_count}.txt"

    if ! pick_ips "${node_count}"; then
        echo "[INFO] ${node_count} nodes (${gpu_count} GPUs) exceeds the node list — stopping"
        break
    fi

    printf "%s\n" "${PICKED[@]}" > "${out_file}"
    if [[ "${IS_GB}" -eq 1 ]]; then
        rack_used=$(( (node_count + USED_TRAYS_PER_RACK - 1) / USED_TRAYS_PER_RACK ))
        echo "[OK] ${out_file} (${node_count} nodes, ${gpu_count} GPUs, ${rack_used} rack(s))"
    else
        echo "[OK] ${out_file} (${node_count} nodes, ${gpu_count} GPUs)"
    fi
    node_count=$(( node_count * 2 ))
done

echo "----------------------------------------------------------------"
echo "[DONE] host lists generated"
