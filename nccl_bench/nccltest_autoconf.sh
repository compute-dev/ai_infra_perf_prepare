#!/bin/bash
# ============================================================================
# nccltest_autoconf.sh — derive the NCCL test parameters for THIS cluster and
#                        print (or write) the exact commands to run.
#
# Run it on one node, outside any container. It is read-only unless you pass
# --apply. The point is that almost every default in the scripts is
# cluster-specific — interface name, HCA list, GID index, image tag, GPU count —
# and guessing them is how multi-node runs fail with LOCALIP=NONE or
# "ibv_modify_qp INIT->RTR ETIMEDOUT".
#
# Usage:
#   bash nccltest_autoconf.sh                 # detect and print commands
#   bash nccltest_autoconf.sh --apply         # also write set_env.sh (backs up any existing one)
#   bash nccltest_autoconf.sh -H hosts.txt    # also check every node is reachable and consistent
#   bash nccltest_autoconf.sh --ssh-port 22   # host ssh port used for the -H check
#   bash nccltest_autoconf.sh --ssh-user root # ssh user for the nodes (default: from ssh config)
#
# Exit codes: 0 ok, 1 bad usage/dependency, 2 detection problem worth reading.
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GID_HELPER="${REPO_DIR}/tools/detect_roce_gid.sh"

APPLY=0
HOSTS_FILE=""
SSH_PORT="${SSH_PORT:-22}"
# Empty means "use whatever user your ~/.ssh/config already specifies".
SSH_USER="${SSH_USER:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --apply)     APPLY=1; shift ;;
        -H|--hosts)  HOSTS_FILE="${2:-}"; shift 2 ;;
        --ssh-port)  SSH_PORT="${2:-}"; shift 2 ;;
        -u|--ssh-user) SSH_USER="${2:-}"; shift 2 ;;
        -h|--help)   sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1 (use -h)" >&2; exit 1 ;;
    esac
done

WARN=0
note()  { echo "[INFO] $*"; }
warn()  { echo "[WARN] $*" >&2; WARN=1; }

echo "=========================================================="
echo " NCCL test parameter detection"
echo "=========================================================="

# ---------------------------------------------------------------------------
# GPUs per node
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found — run this on a GPU node" >&2
    exit 1
fi
GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
if [ "${GPUS_PER_NODE}" -le 0 ]; then
    echo "[ERROR] no GPU detected" >&2
    exit 1
fi
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^ *//')
note "GPUs per node : ${GPUS_PER_NODE}  (${GPU_MODEL})"

# ---------------------------------------------------------------------------
# Bootstrap / TCP interface
# ---------------------------------------------------------------------------
NETWORK_IF="$(ip -o -4 route show default 2>/dev/null | awk '{print $5; exit}')"
if [ -z "${NETWORK_IF}" ]; then
    warn "could not determine the default-route interface; set NETWORK_IF manually"
    NETWORK_IF="bond0"
else
    note "NETWORK_IF    : ${NETWORK_IF}  (won't be auto-detected the same way on every node — verify)"
fi

# ---------------------------------------------------------------------------
# Compute RDMA HCAs (exclude management/storage NICs)
# ---------------------------------------------------------------------------
mapfile -t ALL_HCAS < <(ls /sys/class/infiniband/ 2>/dev/null)
if [ "${#ALL_HCAS[@]}" -eq 0 ]; then
    warn "no RDMA devices found in /sys/class/infiniband — multi-node NCCL will fail"
    NCCL_IB_HCA=""
    GID_INDEX=""
else
    HCCA=()
    for d in "${ALL_HCAS[@]}"; do
        st="$(cat /sys/class/infiniband/$d/ports/1/state 2>/dev/null)"
        [ "${st}" = "4: ACTIVE" ] && HCCA+=("$d")
    done
    note "ACTIVE HCAs   : ${HCCA[*]:-none}"
    note "all HCAs      : ${ALL_HCAS[*]}"

    # Prefer the GPU-NIC affinity result when available
    if [ -f "${SCRIPT_DIR}/gpu_nic_topo.sh" ] && command -v ibdev2netdev >/dev/null 2>&1; then
        NCCL_IB_HCA="$(bash "${SCRIPT_DIR}/gpu_nic_topo.sh" --nccl-hca 2>/dev/null)"
        [ -n "${NCCL_IB_HCA}" ] && note "NCCL_IB_HCA    : ${NCCL_IB_HCA}  (from GPU-PIX/PXB affinity)"
    fi
    if [ -z "${NCCL_IB_HCA:-}" ]; then
        NCCL_IB_HCA="$(IFS=,; echo "${HCCA[*]:-}")"
        [ -n "${NCCL_IB_HCA}" ] && note "NCCL_IB_HCA    : ${NCCL_IB_HCA}  (all ACTIVE HCAs, no affinity filtering)"
    fi
    if [ -z "${NCCL_IB_HCA}" ]; then
        warn "no ACTIVE HCA — check cables and link state before running multi-node tests"
    fi

    # Link layer decides whether a GID index is meaningful at all
    # Probe a device that actually carries NCCL traffic, not just the first
    # ACTIVE one: the ACTIVE list can include a storage/management NIC whose GID
    # table need not match the compute rails'. Prefer a device from the HCA set
    # we just derived.
    FIRST_HCA="$(echo "${NCCL_IB_HCA}" | cut -d, -f1)"
    [ -n "${FIRST_HCA}" ] || FIRST_HCA="${HCCA[0]:-${ALL_HCAS[0]}}"
    LINK_LAYER="$(cat "/sys/class/infiniband/${FIRST_HCA}/ports/1/link_layer" 2>/dev/null || echo unknown)"
    note "fabric        : ${LINK_LAYER}"

    GID_INDEX=""
    if [ "${LINK_LAYER}" = "Ethernet" ]; then
        if [ -x "${GID_HELPER}" ]; then
            if GID_INDEX="$(bash "${GID_HELPER}" "${FIRST_HCA}" 2>/dev/null)"; then
                note "NCCL_IB_GID_INDEX: ${GID_INDEX}  (RoCE v2 + IPv4 on ${FIRST_HCA})"
            else
                warn "no RoCE v2 IPv4 GID on ${FIRST_HCA} — inspect with: bash ${GID_HELPER} ${FIRST_HCA} --table"
                GID_INDEX=""
            fi
        else
            warn "${GID_HELPER} not found; cannot detect the GID index"
        fi
    else
        note "NCCL_IB_GID_INDEX: n/a on InfiniBand (leave unset)"
    fi
fi

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
IMG=""
if command -v docker >/dev/null 2>&1; then
    IMG="$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -iE 'nccltest|nccl-test' | head -1)"
fi
if [ -n "${IMG}" ]; then
    note "IMAGE_NAME    : ${IMG}"
else
    warn "no nccl-test image found locally — build one: docker build -t nccltest:latest ${SCRIPT_DIR}"
    IMG="${IMAGE_NAME:-}"
fi

# ---------------------------------------------------------------------------
# Host list
# ---------------------------------------------------------------------------
HOSTS_FILE="${HOSTS_FILE:-${SCRIPT_DIR}/hosts.txt}"
if [ -f "${HOSTS_FILE}" ]; then
    N="$(grep -vcE '^\s*(#|$)' "${HOSTS_FILE}")"
    note "hosts file    : ${HOSTS_FILE}  (${N} nodes)"
else
    warn "hosts file not found: ${HOSTS_FILE} — copy hosts.txt.example and fill it in"
    N=0
fi

# ---------------------------------------------------------------------------
# Optional reachability check
# ---------------------------------------------------------------------------
SSHX="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p${SSH_PORT}"
# Only force a user when one was given; otherwise the ssh config decides.
[ -n "${SSH_USER}" ] && SSHX="${SSHX} -l ${SSH_USER}"
if [ -n "${HOSTS_FILE}" ] && [ -f "${HOSTS_FILE}" ]; then
    echo ""
    echo "--- SSH reachability (port ${SSH_PORT}) ---"
    UNREACHABLE=0
    while read -r h; do
        [ -z "${h}" ] && continue
        if ssh ${SSHX} -o BatchMode=yes "${h}" true 2>/dev/null; then
            echo "  OK        ${h}"
        else
            echo "  FAIL      ${h}" >&2
            UNREACHABLE=$((UNREACHABLE + 1))
        fi
    done < <(grep -vE '^\s*(#|$)' "${HOSTS_FILE}" | awk '{print $1}')
    [ "${UNREACHABLE}" -gt 0 ] && warn "${UNREACHABLE} node(s) unreachable — set up keyless ssh first (tools/setup_ssh_keyless.sh)"
fi

# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo " Commands to run"
echo "=========================================================="
cat <<EOF

# 0) If set_env.sh does not exist yet, create it from the detected values
#    (or re-run this script with --apply)

# 1) Copy your host list into place, one node address per line
cp hosts.txt.example hosts.txt && vim hosts.txt

# 2) Generate the power-of-two host lists (host8.txt, host16.txt, ...)
bash split_ip.sh
EOF

if [ "${WARN}" -ne 0 ]; then
    echo ""
    echo "#    WARNING: detection was incomplete (see [WARN] above). Fix those first."
fi

cat <<EOF

# 3) Disable PCIe ACS on EVERY node, then reboot (needed for full P2P bandwidth)
#    for h in \$(cat hosts.txt); do ssh -p${SSH_PORT} \$h 'bash -s' < close_acs.sh; done

# 4) Start the container on EVERY node
#    for h in \$(cat hosts.txt); do
#      scp -P${SSH_PORT} launch_container.sh run.sh run_nccltest.sh \\
#                            1node_nccltest.sh parse_nccl_log.py gpu_nic_topo.sh hosts.txt \$h:/root/
#      ssh -p${SSH_PORT} \$h "IMAGE_NAME=${IMG:-<image>} HOSTS_FILE=/root/hosts.txt \\
#           NCCL_SOCKET_IFNAME=${NETWORK_IF} bash /root/launch_container.sh"
#    done

# 5) On ONE node: single-node check first, then the multi-node run
docker exec -i nccltest bash -c "bash /root/1node_nccltest.sh"
docker exec -i nccltest bash -c "HOSTS_FILE=/root/hosts.txt \\
  NCCL_SOCKET_IFNAME=${NETWORK_IF} \\
  ${NCCL_IB_HCA:+NCCL_IB_HCA=${NCCL_IB_HCA} }${GID_INDEX:+NCCL_IB_GID_INDEX=${GID_INDEX} } \\
  bash /root/run_nccltest.sh allreduce all2all allgather reducescatter | tee /root/nccltest_all.log"

# 6) Parse the results (peak OOP busbw, compare against baselines.md)
docker exec -i nccltest python3 /root/parse_nccl_log.py /root/

# To test another scale, just rewrite the host list the container sees and
# re-run step 5 — no container restart needed:
#   cp host16.txt hosts.txt && docker exec -i nccltest bash -c "bash /root/run_nccltest.sh"
EOF

# ---------------------------------------------------------------------------
# --apply
# ---------------------------------------------------------------------------
if [ "${APPLY}" -eq 1 ]; then
    ENV_FILE="${SCRIPT_DIR}/set_env.sh"
    if [ -f "${ENV_FILE}" ]; then
        cp -f "${ENV_FILE}" "${ENV_FILE}.bak.$(date +%Y%m%d%H%M%S)"
        note "backed up existing set_env.sh"
    fi
    cat > "${ENV_FILE}" <<EOF
# Generated by nccltest_autoconf.sh on $(date -Iseconds)
# Review before use — NETWORK_IF in particular must be the same on every node.

# OPTIONAL: a shared directory present at the same path on every node, used to
# keep logs durable. Leave unset if you do not have one.
# export PREFIX_SHARE_PATH="/mnt/data"

export NETWORK_IF="\${NETWORK_IF:-${NETWORK_IF}}"
export GPU_TYPE="\${GPU_TYPE:-${GPU_MODEL:+$(echo "${GPU_MODEL}" | grep -oE '[A-Z][0-9]+' | head -1)}}"
export GPUS_PER_NODE="\${GPUS_PER_NODE:-${GPUS_PER_NODE}}"
export IMAGE_NAME="\${IMAGE_NAME:-${IMG}}"

$( [ -n "${NCCL_IB_HCA:-}" ] && echo "export NCCL_IB_HCA=\"\${NCCL_IB_HCA:-${NCCL_IB_HCA}}\"" || echo "# export NCCL_IB_HCA=\"\"" )
$( [ -n "${GID_INDEX:-}" ] && echo "export NCCL_IB_GID_INDEX=\"\${NCCL_IB_GID_INDEX:-${GID_INDEX}}\"" || echo "# export NCCL_IB_GID_INDEX=\"\"" )
EOF
    note "wrote ${ENV_FILE}"
fi

exit 0
