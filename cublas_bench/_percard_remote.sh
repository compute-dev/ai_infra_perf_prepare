#!/bin/bash
# ============================================================================
# _percard_remote.sh — Per-card cuBLAS GEMM benchmark on a single node
# ----------------------------------------------------------------------------
# Designed to be piped over stdin by cluster_cublas_percard.sh:
#   ssh host 'SHARE_DIR=/path IMAGE=img GPU_TYPE=H800 bash -s' < _percard_remote.sh
#
# It needs no sibling files of its own — only the cublas_bench_<GPU>.sh script
# and cublasMatmulBench_<GPU> binary, which must exist at $SHARE_DIR on the
# remote node (either on a shared filesystem or copied there beforehand).
#
# Environment variables (all passed by the caller):
#   SHARE_DIR       Absolute path to the cublas_bench directory on this node
#                   (default: current directory)
#   IMAGE            Docker image providing the CUDA runtime (required)
#   GPU_TYPE         GPU model name, e.g. H800 (default: H800)
#   GPUS_PER_NODE    Number of GPUs to test (default: auto-detect via nvidia-smi)
#   NODE_ID          Identifier for this node, used as output subdirectory name
#                   (default: hostname)
# ============================================================================
set -uo pipefail

SHARE_DIR="${SHARE_DIR:-$(pwd)}"
GPU_TYPE="${GPU_TYPE:-H800}"
IMAGE="${IMAGE:?IMAGE is required (set -i <image> on the cluster script)}"
NODE_ID="${NODE_ID:-$(hostname -s 2>/dev/null || echo unknown)}"

# Auto-detect GPU count if not explicitly set
if [[ -z "${GPUS_PER_NODE:-}" ]]; then
    GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')
    if [[ -z "$GPUS_PER_NODE" || "$GPUS_PER_NODE" -eq 0 ]]; then
        echo "REMOTE_ERR=no_gpus_detected:$NODE_ID"
        exit 3
    fi
fi

BENCH="cublas_bench_${GPU_TYPE}.sh"
BIN="cublasMatmulBench_${GPU_TYPE}"

# Verify the cublas script and binary exist in SHARE_DIR
[ -f "$SHARE_DIR/$BENCH" ] || { echo "REMOTE_ERR=no_bench_script:$BENCH"; exit 2; }
[ -f "$SHARE_DIR/$BIN" ]   || { echo "REMOTE_ERR=no_binary:$BIN"; exit 2; }

echo "LOCALIP=$NODE_ID"

REL="percard_cluster/$NODE_ID"
OUT="$SHARE_DIR/$REL"
rm -rf "$OUT"; mkdir -p "$OUT"

DEVS=$(seq 0 $((GPUS_PER_NODE-1)) | tr '\n' ' ')

# Per-card loop: cycle CUDA_VISIBLE_DEVICES inside the container and run the
# standard multi-precision GEMM benchmark for each card.
#
# Two things about this invocation:
#   * NO -i / -t. This script is itself piped in over stdin (bash -s), so
#     attaching the container's stdin would let docker consume the rest of the
#     script text. Nothing in here reads stdin, so we detach it explicitly.
#   * stderr is kept and filtered, not discarded. The host's /etc/ld.so.preload
#     leaking `libcompat_fix.so ... cannot be preloaded` lines is noise, but a
#     real CUDA failure ("error while loading shared libraries") must stay in
#     the log — that is what the operator needs to see.
docker run --gpus all --rm --privileged --entrypoint bash \
    -v "$SHARE_DIR":/work -w /work "$IMAGE" -c '
      chmod +x '"$BIN"' 2>/dev/null || true
      for i in '"$DEVS"'; do
        CUDA_VISIBLE_DEVICES=$i bash '"$BENCH"' 2>&1 \
          | grep -vE "libcompat_fix|ld\.so" > '"$REL"'/dev${i}.log
      done
    ' < /dev/null
rc=$?
if [ $rc -ne 0 ]; then echo "REMOTE_ERR=docker_rc_$rc:$NODE_ID"; exit $rc; fi

# Self-check: confirm every card's log has a Gflops line
missing=0
for i in $(seq 0 $((GPUS_PER_NODE-1))); do
  grep -qE 'Gflops[[:space:]]*=' "$OUT/dev${i}.log" 2>/dev/null || missing=$((missing+1))
done
echo "DONE=$NODE_ID files=$(ls "$OUT" | wc -l) missing_gflops=$missing"
