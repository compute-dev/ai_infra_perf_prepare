#!/bin/bash
# ============================================================================
# run_nccltest.sh — multi-node NCCL collective test runner (runs INSIDE the
#                   `nccltest` container on any one of the cluster nodes).
#
# Runs up to four NCCL tests over MPI, one process per GPU:
#     allreduce | all2all | allgather | reducescatter
#
# Usage (inside the container):
#     bash /root/hosts_dir/run_nccltest.sh                     # all four tests
#     bash /root/hosts_dir/run_nccltest.sh allreduce           # one test
#     bash /root/hosts_dir/run_nccltest.sh allreduce all2all   # two tests
#     bash /root/hosts_dir/run_nccltest.sh all2all b=8 e=4G n=20   # override sizes
#
# Required env:
#     NCCL_SOCKET_IFNAME   bootstrap/TCP interface (e.g. bond0, eth0, bond1)
# Optional env (all have defaults):
#     HOSTS_FILE           node list, one IP per line  (default /root/hosts.txt)
#     allips               comma-separated node list, used if HOSTS_FILE is absent
#     NCCL_IB_HCA          RDMA HCA list; auto-detected from GPU-NIC affinity if unset
#     NCCL_IB_GID_INDEX    RoCE v2 GID index; auto-detected if unset (see diag_gid.sh)
#     NCCL_B/NCCL_E/NCCL_F/NCCL_G/NCCL_N   nccl-tests -b/-e/-f/-g/-n defaults
#     PORT                 in-container sshd port used by mpirun (default 7900)
#
# Output: one log per test at /root/nccltest_<test>.log
#         Parse them with: python3 parse_nccl_log.py /root/
# ============================================================================
set -x

# ---------------------------------------------------------------------------
# 1. MPI transport ssh port (must match the port run.sh starts sshd on)
# ---------------------------------------------------------------------------
SSH_PORT="${PORT:-7900}"
PLM_RSH_ARGS="-p ${SSH_PORT}"

# ---------------------------------------------------------------------------
# 2. GPUs per node (auto-detected)
#
#    Count lines starting with "GPU" rather than using `nvidia-smi -L | wc -l`.
#    If the image preloads a spoofing library via /etc/ld.so.preload, `-L` can
#    emit an extra blank line and `wc -l` then reports one GPU too many, which
#    makes mpirun spawn a rank with no visible GPU and abort.
# ---------------------------------------------------------------------------
GPU_PER_NODE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
if [ "${GPU_PER_NODE}" -le 0 ]; then
    echo "[ERROR] No GPU detected — check that nvidia-smi works in this container" >&2
    exit 1
fi
echo "[INFO] GPUs per node: ${GPU_PER_NODE}"

# ---------------------------------------------------------------------------
# 3. Node list
#
#    HOSTS_FILE takes priority over the allips environment variable so the host
#    can edit the file and have the container pick it up without a restart —
#    that is what lets you re-run at 8 / 16 / 32 / 64 nodes from one container.
# ---------------------------------------------------------------------------
HOSTS_FILE="${HOSTS_FILE:-/root/hosts.txt}"
if [ -f "${HOSTS_FILE}" ]; then
    file_ips=$(grep -vE '^\s*(#|$)' "${HOSTS_FILE}" | awk '{print $1}' | paste -sd ',' -)
    if [ -n "${file_ips}" ]; then
        if [ -n "${allips}" ] && [ "${allips}" != "${file_ips}" ]; then
            echo "[INFO] allips differs from ${HOSTS_FILE}; using the file (it can be edited live)"
        fi
        allips="${file_ips}"
        echo "[INFO] Node list read from ${HOSTS_FILE}"
    fi
fi

if [ -z "${allips}" ]; then
    echo "[ERROR] No node list. Do one of:" >&2
    echo "  1) mount a hosts file at /root/hosts.txt (recommended — editable without restart)" >&2
    echo "  2) docker run ... -e allips=ip1,ip2,..." >&2
    exit 1
fi

IFS=',' read -ra IP_ARRAY <<< "${allips}"
NODE_COUNT=${#IP_ARRAY[@]}
TOTAL_GPU=$((NODE_COUNT * GPU_PER_NODE))

# Build the mpirun -H argument: ip1:8,ip2:8,...
HOST_LIST=""
for ip in "${IP_ARRAY[@]}"; do
    if [ -n "${HOST_LIST}" ]; then
        HOST_LIST="${HOST_LIST},"
    fi
    HOST_LIST="${HOST_LIST}${ip}:${GPU_PER_NODE}"
done

echo "[INFO] Nodes: ${NODE_COUNT}, total GPUs: ${TOTAL_GPU}"
echo "[INFO] HOST_LIST: ${HOST_LIST}"

# ---------------------------------------------------------------------------
# 4. NCCL environment
#
#    NCCL_CUMEM_HOST_ENABLE=0 is REQUIRED. On node pairs that can bring up NVLS
#    (NVSwitch in-network reduction), =1 collides with NVLS multicast memory
#    registration and allgather / reducescatter die with
#    "unhandled system error (all_gather.cu:50)". Pairs that cannot bring up
#    NVLS do not crash, which makes this a very misleading failure. =0 is safe
#    on every pair.
# ---------------------------------------------------------------------------
NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"

# ---------------------------------------------------------------------------
# 5. RDMA HCA list
#
#    Priority:
#      (a) NCCL_IB_HCA explicitly set  -> use it
#      (b) gpu_nic_topo.sh             -> pick HCAs that are PIX/PXB-affine to a
#                                         GPU and whose port is Up, which also
#                                         keeps storage/management NICs out
#      (c) ibstat -l                   -> last resort, no affinity filtering
#    Leaving it empty lets NCCL choose, which is usually worse on rail fabrics.
# ---------------------------------------------------------------------------
if [ -z "${NCCL_IB_HCA}" ]; then
    _topo_script="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/gpu_nic_topo.sh"
    if [ -f "${_topo_script}" ]; then
        NCCL_IB_HCA=$(bash "${_topo_script}" --nccl-hca 2>/dev/null)
        [ -n "${NCCL_IB_HCA}" ] && echo "[INFO] NCCL_IB_HCA auto-selected by gpu_nic_topo.sh (GPU-PIX affinity, Up ports)"
    fi
fi
if [ -z "${NCCL_IB_HCA}" ]; then
    if command -v ibstat &>/dev/null; then
        NCCL_IB_HCA=$(ibstat -l 2>/dev/null | paste -sd ',' -)
        [ -n "${NCCL_IB_HCA}" ] && echo "[INFO] NCCL_IB_HCA from ibstat fallback (no affinity filtering)"
    fi
fi
if [ -n "${NCCL_IB_HCA}" ]; then
    echo "[INFO] NCCL_IB_HCA: ${NCCL_IB_HCA}"
else
    echo "[WARN] No RDMA HCA detected — NCCL will choose on its own"
fi

# ---------------------------------------------------------------------------
# 6. Bootstrap interface
#
#    Auto-detect when unset: prefer a bond, else the interface that owns the
#    default route. Getting this wrong makes every rank report LOCALIP=NONE.
# ---------------------------------------------------------------------------
if [ -z "${NCCL_SOCKET_IFNAME}" ]; then
    NCCL_SOCKET_IFNAME=$(ip -o -4 route show default 2>/dev/null | awk '{print $5; exit}')
    if [ -n "${NCCL_SOCKET_IFNAME}" ]; then
        echo "[INFO] NCCL_SOCKET_IFNAME auto-detected from the default route: ${NCCL_SOCKET_IFNAME}"
    fi
fi
if [ -z "${NCCL_SOCKET_IFNAME}" ]; then
    echo "[ERROR] NCCL_SOCKET_IFNAME is not set and could not be auto-detected." >&2
    echo "        Pass it explicitly: NCCL_SOCKET_IFNAME=<iface> bash $0 ..." >&2
    exit 1
fi
echo "[INFO] NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"

# ---------------------------------------------------------------------------
# 7. nccl-tests sizing
#
#    -b: first message size      -e: last message size   -f: step factor
#    -g: GPUs per process        -n: iterations per size
#    Defaults keep -g 1 with mpirun -np = total GPUs: one process per GPU.
#    The reported peak lands at 16G, which is the size the baselines use.
# ---------------------------------------------------------------------------
NCCL_B="${NCCL_B:-8}"
NCCL_E="${NCCL_E:-16G}"
NCCL_F="${NCCL_F:-2}"
NCCL_G="${NCCL_G:-1}"
NCCL_N="${NCCL_N:-50}"

# ---------------------------------------------------------------------------
# Common mpirun runner
# ---------------------------------------------------------------------------
run_nccl_test() {
    local test_name="$1"
    local binary="$2"
    shift 2
    local extra_args="$*"

    # Per-test override of b/e/f/g/n without touching the global defaults
    local b="${TEST_B:-${NCCL_B}}"
    local e="${TEST_E:-${NCCL_E}}"
    local f="${TEST_F:-${NCCL_F}}"
    local g="${TEST_G:-${NCCL_G}}"
    local n="${TEST_N:-${NCCL_N}}"

    echo ""
    echo "==========${test_name}=========="
    echo "[PARAMS] -b ${b} -e ${e} -f ${f} -g ${g} -n ${n}"

    local mpi_cmd=(
        mpirun -np "${TOTAL_GPU}" -H "${HOST_LIST}"
        -v --allow-run-as-root
        # ===== Force MPI onto TCP; disable UCX/UCC/openib =====
        # pml ob1     : ob1 point-to-point layer instead of the default ucx PML
        # btl tcp,self: only TCP and self BTLs, so no openib/vader probing
        # coll ^ucc   : exclude UCC collectives, which otherwise fight NCCL over IB
        --mca pml ob1
        --mca btl tcp,self
        --mca coll ^ucc
        --mca btl_tcp_if_include "${NCCL_SOCKET_IFNAME}"
        --mca oob_tcp_if_include "${NCCL_SOCKET_IFNAME}"
        -x NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
        -x NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE}"
        -x NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION}"
        # ===== large-scale all2all tuning =====
        -x NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-32}"
        -x NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
        # ===== RoCE v2 GID index (commonly 1 or 3 depending on switch config) =====
        -x NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
        # ===== logging =====
        -x NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    )

    if [ -n "${NCCL_IB_HCA}" ]; then
        mpi_cmd+=(-x NCCL_IB_HCA="${NCCL_IB_HCA}")
    fi

    # ===== Only forwarded when explicitly set, so defaults never fight the fabric =====
    [ -n "${UCX_NET_DEVICES}" ]              && mpi_cmd+=(-x UCX_NET_DEVICES="${UCX_NET_DEVICES}")
    [ -n "${NCCL_PROXY_APPEND_BATCH_SIZE}" ] && mpi_cmd+=(-x NCCL_PROXY_APPEND_BATCH_SIZE="${NCCL_PROXY_APPEND_BATCH_SIZE}")
    [ -n "${NCCL_BUFFSIZE}" ]                && mpi_cmd+=(-x NCCL_BUFFSIZE="${NCCL_BUFFSIZE}")
    [ -n "${NCCL_NET_GDR_LEVEL}" ]           && mpi_cmd+=(-x NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}")
    [ -n "${NCCL_SHM_DISABLE}" ]             && mpi_cmd+=(-x NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE}")
    [ -n "${NCCL_P2P_LEVEL}" ]               && mpi_cmd+=(-x NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL}")
    [ -n "${NCCL_IB_TIMEOUT}" ]              && mpi_cmd+=(-x NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT}")
    [ -n "${NCCL_IB_RETRY_CNT}" ]            && mpi_cmd+=(-x NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT}")
    [ -n "${NCCL_DEBUG_SUBSYS}" ]            && mpi_cmd+=(-x NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS}")
    [ -n "${NCCL_DEBUG_FILE}" ]              && mpi_cmd+=(-x NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE}")

    mpi_cmd+=(
        -mca plm_rsh_args "${PLM_RSH_ARGS}"
        "${binary}" -b "${b}" -e "${e}" -f "${f}" -g "${g}" -n "${n}" ${extra_args}
    )

    echo "[CMD] ${mpi_cmd[*]}"
    echo "[CMD] ${mpi_cmd[*]}" | tee "/root/nccltest_${test_name}.log"
    "${mpi_cmd[@]}" 2>&1 | tee -a "/root/nccltest_${test_name}.log"
    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ]; then
        echo "[${test_name}] OK"
    else
        echo "[${test_name}] FAILED (exit_code=${exit_code})"
    fi

    # Cool-down between tests
    sleep 3
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
NCCL_TEST_DIR="${NCCL_TEST_DIR:-/root/nccl-tests}"

ALL_TESTS="allreduce all2all allgather reducescatter"

# Argument parsing:
#   - no positional args      -> run all four tests
#   - positional test names   -> run just those
#   - key=value (b/e/f/g/n)   -> override nccl-tests sizing for this invocation
SELECTED_TESTS=""
for arg in "$@"; do
    if [[ "${arg}" == *=* ]]; then
        key="${arg%%=*}"
        val="${arg#*=}"
        case "${key,,}" in
            b) NCCL_B="${val}" ;;
            e) NCCL_E="${val}" ;;
            f) NCCL_F="${val}" ;;
            g) NCCL_G="${val}" ;;
            n) NCCL_N="${val}" ;;
            *)
                echo "[ERROR] Unknown key: ${key} (supported: b/e/f/g/n)" >&2
                exit 1
                ;;
        esac
    else
        SELECTED_TESTS="${SELECTED_TESTS} ${arg}"
    fi
done

SELECTED_TESTS="$(echo "${SELECTED_TESTS}" | xargs)"
if [ -z "${SELECTED_TESTS}" ]; then
    SELECTED_TESTS="${ALL_TESTS}"
fi

for t in ${SELECTED_TESTS}; do
    case "${t}" in
        allreduce|all2all|allgather|reducescatter) ;;
        *)
            echo "[ERROR] Unknown test: ${t}" >&2
            echo "[USAGE] bash $0 [allreduce|all2all|allgather|reducescatter ...] [b=8] [e=16G] [f=2] [g=1] [n=50]" >&2
            echo "        No test name runs all four." >&2
            echo "        Defaults can also come from NCCL_B/NCCL_E/NCCL_F/NCCL_G/NCCL_N." >&2
            exit 1
            ;;
    esac
done

echo "[INFO] Tests to run: ${SELECTED_TESTS}"
echo "[INFO] nccl-tests params: -b ${NCCL_B} -e ${NCCL_E} -f ${NCCL_F} -g ${NCCL_G} -n ${NCCL_N}"

for t in ${SELECTED_TESTS}; do
    case "${t}" in
        allreduce)
            run_nccl_test "allreduce" "${NCCL_TEST_DIR}/build/all_reduce_perf_mpi"
            ;;
        all2all)
            run_nccl_test "all2all" "${NCCL_TEST_DIR}/build/alltoall_perf_mpi"
            ;;
        allgather)
            run_nccl_test "allgather" "${NCCL_TEST_DIR}/build/all_gather_perf_mpi"
            ;;
        reducescatter)
            run_nccl_test "reducescatter" "${NCCL_TEST_DIR}/build/reduce_scatter_perf_mpi"
            ;;
    esac
done
