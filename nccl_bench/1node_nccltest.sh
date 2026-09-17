#!/bin/bash
# ============================================================================
# 1node_nccltest.sh — single-node NCCL test (intra-node NVLink only).
#
# This is the quickest sanity check: it exercises the NVLink/NVSwitch domain of
# one node with no MPI and no network. Use it before the multi-node run so that
# a node-local interconnect problem is not confused with a fabric problem.
#
# Note the difference from the multi-node run: here ONE process drives all GPUs
# (-g = GPUs per node), so there is no mpirun and no per-GPU rank.
#
# Usage (inside the container):
#     bash /root/hosts_dir/1node_nccltest.sh            # all four tests
#     bash /root/hosts_dir/1node_nccltest.sh allreduce  # one test
#
# Output: /root/1node_nccltest_<test>.log
# ============================================================================
set -u

NCCL_TEST_DIR="${NCCL_TEST_DIR:-/root/nccl-tests}"
B="${B:-8}"
E="${E:-16G}"
F="${F:-2}"
N="${N:-50}"

# Number of local GPUs, and the device list that goes with it
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
if [ "${GPU_COUNT}" -le 0 ]; then
    echo "[ERROR] No GPU detected — check nvidia-smi" >&2
    exit 1
fi
DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
echo "[INFO] ${GPU_COUNT} GPUs: CUDA_VISIBLE_DEVICES=${DEVICES}"

ALL_TESTS="allreduce all2all allgather reducescatter"
SELECTED="${*:-${ALL_TESTS}}"

# Map a test name to its non-MPI binary and its log suffix
run_one() {
    local name="$1" bin="$2"
    echo ""
    echo "========== 1-node ${name} =========="
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${DEVICES}" "${NCCL_TEST_DIR}/build/${bin}" \
        -b "${B}" -e "${E}" -f "${F}" -g "${GPU_COUNT}" -n "${N}" \
        | tee "/root/1node_nccltest_${name}.log"
    echo "[${name}] exit=${PIPESTATUS[0]}"
    sleep 5
}

for t in ${SELECTED}; do
    case "${t}" in
        allreduce)     run_one allreduce     all_reduce_perf ;;
        all2all)       run_one all2all       alltoall_perf ;;
        allgather)     run_one allgather     all_gather_perf ;;
        reducescatter) run_one reducescatter reduce_scatter_perf ;;
        *)
            echo "[ERROR] Unknown test: ${t}" >&2
            echo "[USAGE] bash $0 [allreduce|all2all|allgather|reducescatter ...]" >&2
            exit 1
            ;;
    esac
done

echo ""
echo "[DONE] Logs: /root/1node_nccltest_*.log"
echo "       Parse with: python3 /root/parse_nccl_log.py /root/"
