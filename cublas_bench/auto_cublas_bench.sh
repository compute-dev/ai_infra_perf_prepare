#!/bin/bash
# cuBLAS GEMM automated benchmark entry point
#   - Selects the per-GPU test script based on the GPU_NAME env var
#   - Tees output to cublas_bench_<GPU>.log
#   - Parses measured TFLOPS and compares against cublasMatmulBenchBaseline.yaml
#
# Usage:
#   GPU_NAME=B200 bash auto_cublas_bench.sh
#
# Supported GPU_NAME (case-insensitive):
#   GB300, GB200, B300, B200, H200, H100, H800, A800, H20, L20

set -eo pipefail

# cd to the script directory so relative paths work regardless of caller cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 1. validate GPU_NAME ----------
if [[ -z "${GPU_NAME:-}" ]]; then
    echo "Error: GPU_NAME environment variable is not set" >&2
    echo "Usage: GPU_NAME=<model> bash $(basename "$0")" >&2
    echo "Supported models: GB300 GB200 B300 B200 H200 H100 H800 A800 H20 L20" >&2
    exit 2
fi

# Normalise to upper case for matching
GPU_UPPER="$(echo "$GPU_NAME" | tr '[:lower:]' '[:upper:]')"

# ---------- 2. select test script and baseline GPU ----------
# BENCH_SCRIPT : the cublas_bench_*.sh to execute
# BASELINE_GPU : GPU model passed to parse_cublas_simple.py (for YAML baseline lookup)
case "$GPU_UPPER" in
    B200)
        BENCH_SCRIPT="cublas_bench_B200.sh";      BASELINE_GPU="B200"  ;;
    B300)
        BENCH_SCRIPT="cublas_bench_B300.sh";      BASELINE_GPU="B300"  ;;
    GB200)
        BENCH_SCRIPT="cublas_bench_GB200.sh";     BASELINE_GPU="GB200" ;;
    GB300)
        BENCH_SCRIPT="cublas_bench_GB300.sh";     BASELINE_GPU="GB300" ;;
    H100)
        BENCH_SCRIPT="cublas_bench_H100.sh"; BASELINE_GPU="H100"  ;;
    H200)
        BENCH_SCRIPT="cublas_bench_H200.sh"; BASELINE_GPU="H200"  ;;
    H800)
        BENCH_SCRIPT="cublas_bench_H800.sh";     BASELINE_GPU="H800"  ;;
    A800)
        BENCH_SCRIPT="cublas_bench_A800.sh";     BASELINE_GPU="A800"  ;;
    H20)
        BENCH_SCRIPT="cublas_bench_H20.sh";       BASELINE_GPU="H20"   ;;
    L20)
        BENCH_SCRIPT="cublas_bench_L20.sh";       BASELINE_GPU="L20"   ;;
    *)
        echo "Error: unsupported GPU_NAME='$GPU_NAME'" >&2
        echo "Supported models: GB300 GB200 B300 B200 H200 H100 H800 A800 H20 L20" >&2
        exit 2
        ;;
esac

# ---------- 3. file existence pre-checks ----------
if [[ ! -f "$BENCH_SCRIPT" ]]; then
    echo "Error: test script not found: $BENCH_SCRIPT" >&2
    exit 2
fi
if [[ ! -f "cublasMatmulBenchBaseline.yaml" ]]; then
    echo "Error: baseline file not found: cublasMatmulBenchBaseline.yaml" >&2
    exit 2
fi
if [[ ! -f "parse_cublas_simple.py" ]]; then
    echo "Error: parser script not found: parse_cublas_simple.py" >&2
    exit 2
fi

LOG_FILE="cublas_bench_${GPU_UPPER}.log"

echo "========================================================"
echo "  cuBLAS GEMM Benchmark"
echo "  GPU_NAME      : $GPU_UPPER"
echo "  Test script   : $BENCH_SCRIPT"
echo "  Baseline GPU  : $BASELINE_GPU (cublasMatmulBenchBaseline.yaml)"
echo "  Log file      : $LOG_FILE"
echo "========================================================"

# ---------- 4. pre-flight: check the cublasMatmulBench_<XXX> binary can load all shared libs ----------
# Identify the actual binary called by BENCH_SCRIPT (e.g. ./cublasMatmulBench_H20)
BENCH_BIN=$(grep -oE '\./cublasMatmulBench_[A-Za-z0-9_]+' "$BENCH_SCRIPT" | head -1)
if [[ -n "$BENCH_BIN" ]]; then
    BENCH_BIN_PATH="${BENCH_BIN#./}"
    if [[ ! -f "$BENCH_BIN_PATH" ]]; then
        echo "Error: test binary not found: $BENCH_BIN_PATH" >&2
        exit 2
    fi
    chmod +x "$BENCH_BIN_PATH" 2>/dev/null || true
    if command -v ldd >/dev/null 2>&1; then
        MISSING=$(ldd "$BENCH_BIN_PATH" 2>&1 | grep 'not found' || true)
        if [[ -n "$MISSING" ]]; then
            echo "Error: $BENCH_BIN_PATH has missing shared-library dependencies:" >&2
            echo "$MISSING" >&2
            echo "" >&2
            echo "Hint: this usually means the CUDA runtime is not configured. Try:" >&2
            echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}" >&2
            echo "Or run inside an NVIDIA official container (e.g. nvcr.io/nvidia/pytorch:25.10-py3)" >&2
            echo "Diagnostics: run nvidia-smi to confirm driver/CUDA version; check LD_LIBRARY_PATH includes CUDA lib64" >&2
            exit 3
        fi
    fi
fi

# ---------- 5. run benchmark + live parse ----------
# Use PIPESTATUS to capture the real exit code of each pipeline stage
# Note: PIPESTATUS is overwritten by the next command, so copy it immediately
bash "$BENCH_SCRIPT" 2>&1 | tee "$LOG_FILE" | python3 parse_cublas_simple.py - "$BASELINE_GPU"
RC_ARRAY=("${PIPESTATUS[@]}")
BENCH_RC=${RC_ARRAY[0]:-0}
PARSE_RC=${RC_ARRAY[2]:-0}

# Fallback: even if the ldd pre-flight missed it, scan the log for "shared libraries" errors
if grep -q 'error while loading shared libraries' "$LOG_FILE" 2>/dev/null; then
    echo "" >&2
    echo "Error: 'shared library loading failure' detected in the test log; the benchmark did not actually run" >&2
    echo "  See the 'error while loading shared libraries' line(s) in $LOG_FILE" >&2
    echo "  Diagnostics: run nvidia-smi to confirm driver state; check LD_LIBRARY_PATH includes CUDA lib64" >&2
    exit 3
fi

if [[ $BENCH_RC -ne 0 ]]; then
    echo "Error: test script $BENCH_SCRIPT failed (rc=$BENCH_RC)" >&2
    exit "$BENCH_RC"
fi

exit "$PARSE_RC"
