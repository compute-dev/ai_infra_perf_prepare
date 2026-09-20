#!/bin/bash
# ============================================================================
# detect_roce_gid.sh — find the RoCE v2 + IPv4 GID index of an RDMA device.
#
# Single node, read-only, no ssh. This is the primitive behind
# NCCL_IB_GID_INDEX. Getting it wrong is a classic multi-node failure: every
# QP fails to connect with "ibv_modify_qp INIT->RTR ETIMEDOUT", which looks
# like a fabric fault but is really just the wrong GID.
#
# Do NOT hardcode 3 across clusters. The index is assigned by the driver from
# the port's GID table and shifts when the IP scheme or RoCE version changes.
#
# Usage:
#   bash detect_roce_gid.sh <dev>        # print the index, e.g. 3
#   bash detect_roce_gid.sh <dev> --table   # print the whole GID table
#   bash detect_roce_gid.sh --first      # pick the first ACTIVE device
#
# Exit code 1 if no RoCE v2 IPv4 GID is found (check `show_gids` and, for
# InfiniBand, note that IB fabrics do not use GID indices at all).
# ============================================================================

set -uo pipefail

show_table() {
    local dev="$1"
    printf "%-6s %-12s %-10s %s\n" "IDX" "TYPE" "NETDEV" "GID"
    printf "%-6s %-12s %-10s %s\n" "---" "----" "------" "---"
    local idx type netdev gid
    for idx in $(seq 0 15); do
        local base="/sys/class/infiniband/${dev}/ports/1"
        [ -r "${base}/gids/${idx}" ] || continue
        gid="$(cat "${base}/gids/${idx}" 2>/dev/null)"
        [ -z "${gid}" ] && continue
        # Skip the all-zero entry
        [ "${gid}" = "0000:0000:0000:0000:0000:0000:0000:0000" ] && continue
        type="$(cat "${base}/gid_attrs/types/${idx}" 2>/dev/null || echo "?")"
        netdev="$(cat "${base}/gid_attrs/ndevs/${idx}" 2>/dev/null || echo "-")"
        printf "%-6s %-12s %-10s %s\n" "${idx}" "${type}" "${netdev}" "${gid}"
    done
}

# Parse the GID table and return the index of the first RoCE v2 entry that
# carries an IPv4-mapped address (::ffff:a.b.c.d).
detect_index() {
    local dev="$1"
    local idx type gid
    for idx in $(seq 0 15); do
        local base="/sys/class/infiniband/${dev}/ports/1"
        [ -r "${base}/gids/${idx}" ] || continue
        type="$(cat "${base}/gid_attrs/types/${idx}" 2>/dev/null)"
        gid="$(cat "${base}/gids/${idx}" 2>/dev/null)"
        # RoCE v2 with an IPv4 address looks like 0000:...:ffff:0a00:0001
        if [[ "${type}" == *v2* ]] && [[ "${gid}" == *ffff:* ]]; then
            echo "${idx}"
            return 0
        fi
    done
    return 1
}

# ---------------- arguments ----------------
DEV="${1:-}"
if [ -z "${DEV}" ] || [ "${DEV}" = "--first" ]; then
    DEV="$(ls /sys/class/infiniband/ 2>/dev/null | while read -r d; do
        st="$(cat /sys/class/infiniband/$d/ports/1/state 2>/dev/null)"
        [ "${st}" = "4: ACTIVE" ] && { echo "$d"; break; }
    done)"
fi

if [ -z "${DEV}" ]; then
    echo "[ERROR] no RDMA device found (looked in /sys/class/infiniband)" >&2
    exit 1
fi
if [ ! -d "/sys/class/infiniband/${DEV}" ]; then
    echo "[ERROR] no such RDMA device: ${DEV}" >&2
    echo "        available: $(ls /sys/class/infiniband/ 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
fi

# InfiniBand and RoCE are handled differently — check the link layer first
LINK_LAYER="$(cat "/sys/class/infiniband/${DEV}/ports/1/link_layer" 2>/dev/null || echo unknown)"
DEV_RATE="$(cat "/sys/class/infiniband/${DEV}/ports/1/rate" 2>/dev/null || echo unknown)"
DEV_STATE="$(cat "/sys/class/infiniband/${DEV}/ports/1/state" 2>/dev/null || echo unknown)"
echo "[INFO] device=${DEV} link_layer=${LINK_LAYER} rate=${DEV_RATE} state=${DEV_STATE}" >&2

case "${2:-}" in
    --table)
        show_table "${DEV}"
        exit 0
        ;;
    -h|--help)
        # This script is often piped to a remote `bash -s`, where $0 is just
        # "bash" and cannot be re-read. Fall back to inline usage text.
        if [ -r "$0" ] && [ "$0" != "bash" ]; then
            sed -n '2,20p' "$0"
        else
            echo "usage: detect_roce_gid.sh <dev>          # print the RoCE v2 + IPv4 GID index"
            echo "       detect_roce_gid.sh <dev> --table  # dump the whole GID table"
            echo "       detect_roce_gid.sh --first        # use the first ACTIVE device"
        fi
        exit 0
        ;;
esac

if [ "${LINK_LAYER}" = "InfiniBand" ]; then
    echo "[INFO] ${DEV} is InfiniBand — GID index does not apply, leave NCCL_IB_GID_INDEX unset." >&2
    exit 2
fi

if ! IDX="$(detect_index "${DEV}")"; then
    echo "[ERROR] no RoCE v2 IPv4 GID on ${DEV}." >&2
    echo "        GID table:" >&2
    show_table "${DEV}" >&2
    echo "        Check that the port has an IPv4 address and that RoCE v2 is enabled." >&2
    exit 1
fi

echo "${IDX}"
