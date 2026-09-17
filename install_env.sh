#!/bin/bash
# ============================================================================
# install_env.sh — install what this repository actually needs, on the machine
#                  you drive the tests from (the "control machine").
#
# The four gates are plain shell + python3 + ssh + docker. There is no conda
# environment, no pssh, no object-storage client and no Python package to
# install. This script only fills in OS-level gaps.
#
# It is safe to re-run. It does not modify sshd configuration or any keys.
#
# Usage:
#     sudo bash install_env.sh
# ============================================================================
set -e

echo "=== checking prerequisites for ai_infra_perf_prepare ==="

# Detect the package manager
if command -v apt-get >/dev/null 2>&1; then
    PM="apt-get"
    PM_INSTALL="apt-get install -y"
elif command -v dnf >/dev/null 2>&1; then
    PM="dnf"
    PM_INSTALL="dnf install -y"
elif command -v yum >/dev/null 2>&1; then
    PM="yum"
    PM_INSTALL="yum install -y"
else
    echo "[ERROR] no supported package manager (apt-get / dnf / yum)" >&2
    exit 1
fi
echo "[INFO] package manager: ${PM}"

# What each gate actually needs on the control machine:
#   bash, ssh, scp, rsync  - transport to the nodes
#   sshpass                - only for the one-time key push (tools/setup_ssh_keyless.sh)
#   python3                - all parsers and the summariser
#   tar, gzip              - tools/collect_results.sh
#   git, git-lfs           - this repo tracks large .so files with git-lfs
#   curl, wget             - fetching images / general use
PKGS="bash openssh-client rsync sshpass python3 tar gzip curl wget git ca-certificates"

if [ "${PM}" = "apt-get" ]; then
    apt-get update
fi

# shellcheck disable=SC2086
${PM_INSTALL} ${PKGS} || {
    echo "[WARN] some packages failed to install; continuing to check what is present" >&2
}

# git-lfs is packaged separately on most distros
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "[INFO] installing git-lfs"
    # shellcheck disable=SC2086
    ${PM_INSTALL} git-lfs 2>/dev/null || {
        echo "[WARN] git-lfs not available from the package manager." >&2
        echo "       cublas_bench/GB300/gemm-memread/ is stored in git-lfs;" >&2
        echo "       without it you will get pointer files instead of libraries." >&2
        echo "       See https://git-lfs.com for manual install steps." >&2
    }
fi

echo
echo "=== result ==="
MISSING=0
for c in ssh scp rsync sshpass python3 tar git; do
    if command -v "${c}" >/dev/null 2>&1; then
        printf "  %-10s %s\n" "${c}" "$(command -v "${c}")"
    else
        printf "  %-10s MISSING\n" "${c}"
        MISSING=$((MISSING + 1))
    fi
done
if command -v git-lfs >/dev/null 2>&1; then
    printf "  %-10s %s\n" "git-lfs" "$(command -v git-lfs)"
else
    printf "  %-10s MISSING  (needed to check out the GEMM libraries)\n" "git-lfs"
    MISSING=$((MISSING + 1))
fi

if [ "${MISSING}" -gt 0 ]; then
    echo
    echo "[WARN] ${MISSING} prerequisite(s) still missing."
    exit 1
fi

echo
echo "[OK] control machine ready."
echo
echo "Next: set up passwordless ssh to the nodes, then run gate 0."
echo "  cp preflight/hosts.txt.example preflight/hosts.txt && vim preflight/hosts.txt"
echo "  NODE_PASSWORD='...' bash tools/setup_ssh_keyless.sh preflight/hosts.txt --port 22"
echo "  cd preflight && bash cluster_runtime_check.sh -H hosts.txt -P 22 -o ../results/preflight/runtime.csv"
