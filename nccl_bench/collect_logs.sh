#!/bin/bash
# ============================================================================
# collect_logs.sh — copy the NCCL test logs out of the container so they land
#                   in the results tree alongside the other gates.
#
# Run this on the node where you ran run_nccltest.sh / 1node_nccltest.sh.
# The logs live inside the container at /root/; they are useless to
# tools/collect_results.sh until they are on the host.
#
# Usage:
#     bash collect_logs.sh
#     bash collect_logs.sh --container nccltest --out ../results/nccl
#
# Options:
#     --container NAME   container name (default: nccltest)
#     --out DIR          output directory (default: <repo>/results/nccl)
#
# Copies: nccltest_<op>.log (multi-node) and 1node_nccltest_<op>.log (single-node)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="nccltest"
OUT="${SCRIPT_DIR}/../results/nccl"

while [ $# -gt 0 ]; do
    case "$1" in
        --container) CONTAINER="${2:-}"; shift 2 ;;
        --out)       OUT="${2:-}"; shift 2 ;;
        -h|--help)   sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown option: $1 (use --help)" >&2; exit 1 ;;
    esac
done

if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
    echo "[ERROR] no such container: ${CONTAINER}" >&2
    echo "        Run this on the node where the tests ran, or pass --container NAME." >&2
    exit 1
fi

mkdir -p "${OUT}"

# List the logs inside the container first: copying a glob that matches nothing
# would otherwise look like a silent success.
mapfile -t LOGS < <(docker exec -i "${CONTAINER}" bash -c \
    'ls /root/nccltest_*.log /root/1node_nccltest_*.log 2>/dev/null')

if [ "${#LOGS[@]}" -eq 0 ]; then
    echo "[ERROR] no nccl test logs inside ${CONTAINER}." >&2
    echo "        Expected /root/nccltest_*.log or /root/1node_nccltest_*.log." >&2
    echo "        Did the tests actually run in this container?" >&2
    exit 1
fi

echo "[INFO] found ${#LOGS[@]} log(s) in ${CONTAINER}"
for f in "${LOGS[@]}"; do
    docker cp "${CONTAINER}:${f}" "${OUT}/" >/dev/null 2>&1 \
        && echo "  copied $(basename "$f")" \
        || echo "  FAILED $(basename "$f")" >&2
done

echo "[INFO] written to ${OUT}"
ls -1 "${OUT}" | sed 's/^/  /'
