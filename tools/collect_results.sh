#!/bin/bash
# ============================================================================
# collect_results.sh — package everything the four gates produced into one
#                      tarball for submission.
#
# It checks each gate's output directory, tells you what is missing, writes an
# environment snapshot of this machine, and tars the lot.
#
# Usage:
#     bash collect_results.sh --vendor <name> --gpu <model> [--dir results] [-o out.tar.gz]
#
# Options:
#     --dir DIR        results root, default ./results
#     --vendor NAME    vendor name, used in the file name and the summary
#     --gpu MODEL      GPU model, used in the file name and the summary
#     -o FILE          output tarball (default results_<vendor>_<gpu>_<date>.tar.gz)
#     --allow-partial  package even if some gates have no output
#     --skip-snapshot  do not collect the local environment snapshot
#
# Expected layout under --dir (each gate writes its own subdirectory):
#
#     results/
#       preflight/   runtime_*.csv, spec_*.csv
#       cublas/      cublas_bench_<GPU>.log and/or percard_cluster/<ip>/dev*.log
#       nccl/        nccltest_*.log, 1node_nccltest_*.log
#       perftest/    perftest_*.log, summary_*.txt
#
# Any gate with no files is reported; it is not silently skipped.
# ============================================================================
set -uo pipefail

RESULTS_DIR="results"
VENDOR=""
GPU=""
OUT=""
ALLOW_PARTIAL=0
SKIP_SNAPSHOT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dir)            RESULTS_DIR="${2:-}"; shift 2 ;;
        --vendor)         VENDOR="${2:-}"; shift 2 ;;
        --gpu)            GPU="${2:-}"; shift 2 ;;
        -o|--output)      OUT="${2:-}"; shift 2 ;;
        --allow-partial)  ALLOW_PARTIAL=1; shift ;;
        --skip-snapshot)  SKIP_SNAPSHOT=1; shift ;;
        -h|--help)        sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown option: $1 (use --help)" >&2; exit 1 ;;
    esac
done

[ -n "${VENDOR}" ] || { echo "[ERROR] --vendor is required (goes into the file name and summary)" >&2; exit 1; }
[ -n "${GPU}" ]    || { echo "[ERROR] --gpu is required" >&2; exit 1; }

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "[ERROR] results directory not found: ${RESULTS_DIR}" >&2
    echo "        Run the gates first; they write into ${RESULTS_DIR}/<gate>/." >&2
    exit 1
fi

# Sanitise the vendor name for use in a file name.
# printf, not echo: echo appends a newline, and `tr -c` would turn that
# trailing newline into a stray underscore.
SAFE_VENDOR="$(printf '%s' "${VENDOR}" | tr -c 'A-Za-z0-9._-' '_')"
TS="$(date +%Y%m%d)"
[ -n "${OUT}" ] || OUT="results_${SAFE_VENDOR}_${GPU}_${TS}.tar.gz"

# ---------------------------------------------------------------------------
# Check what each gate produced
# ---------------------------------------------------------------------------
echo "===== checking gate outputs under ${RESULTS_DIR}/ ====="

declare -A GATE_DESC=(
    [preflight]="node-consistency preflight (runtime_*.csv, spec_*.csv)"
    [cublas]="cuBLAS single-GPU benchmark (cublas_bench_*.log, percard_cluster/)"
    [nccl]="NCCL collectives (nccltest_*.log)"
    [perftest]="RDMA point-to-point (summary_*.txt, perftest_*.log)"
)
GATE_ORDER=(preflight cublas nccl perftest)

MISSING=()
TOTAL_FILES=0
for g in "${GATE_ORDER[@]}"; do
    d="${RESULTS_DIR}/${g}"
    if [ ! -d "${d}" ]; then
        MISSING+=("${g}")
        printf "  %-10s MISSING  (no directory)          %s\n" "${g}" "${GATE_DESC[${g}]}"
        continue
    fi
    n=$(find "${d}" -type f | wc -l)
    TOTAL_FILES=$((TOTAL_FILES + n))
    if [ "${n}" -eq 0 ]; then
        MISSING+=("${g}")
        printf "  %-10s EMPTY                          %s\n" "${g}" "${GATE_DESC[${g}]}"
    else
        sz=$(du -sh "${d}" 2>/dev/null | awk '{print $1}')
        printf "  %-10s OK       %4d file(s), %6s  %s\n" "${g}" "${n}" "${sz}" "${GATE_DESC[${g}]}"
    fi
done

echo
echo "total: ${TOTAL_FILES} file(s)"
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo
    echo "[WARN] no output for: ${MISSING[*]}"
    echo "       A gate with no output has not been accepted — it is not evidence."
    echo "       Re-run it, or pass --allow-partial if you intend to submit an"
    echo "       incomplete package (the summary will record what is missing)." >&2
    if [ "${ALLOW_PARTIAL}" -eq 0 ]; then
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# Environment snapshot — makes the numbers interpretable later
# ---------------------------------------------------------------------------
if [ "${SKIP_SNAPSHOT}" -eq 0 ]; then
    SNAP="${RESULTS_DIR}/environment.txt"
    {
        echo "collected_at : $(date -Iseconds)"
        echo "collected_on : $(hostname)"
        echo "vendor       : ${VENDOR}"
        echo "gpu_reported : ${GPU}"
        echo "--- hardware ---"
        echo "gpu_model    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^ *//' || true)"
        # `grep -c` prints 0 AND exits 1 when nothing matches, so a plain
        # `|| echo 0` would print the count twice. `|| true` is what we want.
        echo "gpu_count    : $(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
        echo "cpu          : $(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"
        echo "memory       : $(awk '/MemTotal/{printf "%.0f GB\n", $2/1024/1024}' /proc/meminfo 2>/dev/null)"
        echo "os           : $(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2- | tr -d '\"')"
        echo "kernel       : $(uname -r)"
        echo "--- software ---"
        echo "driver       : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        echo "cuda_api     : $(nvidia-smi 2>/dev/null | awk -F'CUDA Version:' '/CUDA Version/{print $2}' | awk '{print $1}')"
        echo "docker       : $(docker --version 2>/dev/null | head -1)"
        # Only the count. The exact image each gate used is recorded in that
        # gate's own logs, and listing every tag here would bury the report in
        # unrelated registry names.
        echo "image_count  : $(docker images -q 2>/dev/null | wc -l)"
        echo "--- fabric ---"
        echo "hcas         : $(ls /sys/class/infiniband/ 2>/dev/null | tr '\n' ' ')"
        echo "ofed         : $(ofed_info -s 2>/dev/null | head -1)"
    } > "${SNAP}" 2>/dev/null
    echo "[INFO] wrote ${SNAP}"
fi

# ---------------------------------------------------------------------------
# Manifest + package
# ---------------------------------------------------------------------------
MANIFEST="${RESULTS_DIR}/MANIFEST.txt"
{
    echo "ai_infra_perf_prepare results package"
    echo "vendor  : ${VENDOR}"
    echo "gpu     : ${GPU}"
    echo "created : $(date -Iseconds)"
    echo "host    : $(hostname)"
    echo ""
    echo "gates present : $(for g in "${GATE_ORDER[@]}"; do [ -d "${RESULTS_DIR}/${g}" ] && [ "$(find "${RESULTS_DIR}/${g}" -type f | wc -l)" -gt 0 ] && echo -n "${g} "; done)"
    echo "gates missing : ${MISSING[*]:-none}"
    echo ""
    echo "file listing:"
    ( cd "${RESULTS_DIR}" && find . -type f | sort | sed 's|^\./|  |' )
} > "${MANIFEST}"

echo "[INFO] wrote ${MANIFEST}"

# Package from the parent so the tarball unpacks to a named top-level dir
PARENT="$(dirname "$(cd "${RESULTS_DIR}" && pwd)")"
BASE="$(basename "$(cd "${RESULTS_DIR}" && pwd)")"

echo "[INFO] creating ${OUT}"
tar -czf "${OUT}" -C "${PARENT}" "${BASE}" || {
    echo "[ERROR] tar failed" >&2
    exit 1
}

SIZE="$(du -h "${OUT}" | awk '{print $1}')"
echo
echo "===== package ready ====="
echo "  ${OUT}  (${SIZE})"
echo
echo "Contents: the four gates' raw logs, the environment snapshot and MANIFEST.txt."
echo "Generate a written summary from it with:"
echo "    python3 tools/summarize.py ${RESULTS_DIR} --vendor ${VENDOR} --gpu ${GPU}"
