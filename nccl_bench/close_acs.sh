#!/bin/bash
# ============================================================================
# close_acs.sh — disable PCIe ACS (Access Control Services) on every PCI device.
#
# Why: with ACS enabled, the PCIe switch may refuse peer-to-peer DMA between
# devices under the same switch. That breaks GPUDirect P2P and costs you
# intra-node NCCL bandwidth. Run this on every node BEFORE the NCCL tests.
#
# Requirements: root, lspci, setpci, dmidecode (pciutils / dmidecode packages).
# This is a host-level change and usually needs a reboot to be fully effective.
#
# Usage, on each node:
#     sudo bash close_acs.sh
#
# Verify afterwards:
#     for BDF in $(lspci -d "*:*:*" | awk '{print $1}'); do
#         setpci -s $BDF ECAP_ACS+0x6.w 2>/dev/null
#     done
#   Every value that prints should be 0x0000.
# ============================================================================

PLATFORM=$(dmidecode --string system-product-name 2>/dev/null)
logger "PLATFORM=${PLATFORM}"
echo "[INFO] Platform: ${PLATFORM}"

# Extended PCI config space needs root
if [ "$EUID" -ne 0 ]; then
  echo "ERROR: $0 must be run as root" >&2
  exit 1
fi

CHANGED=0
SKIPPED=0
FAILED=0

for BDF in $(lspci -d "*:*:*" | awk '{print $1}'); do

    # Skip devices that have no ACS capability
    setpci -v -s ${BDF} ECAP_ACS+0x6.w > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    setpci -v -s ${BDF} ECAP_ACS+0x6.w=0x0000
    if [ $? -ne 0 ]; then
        logger "Error disabling ACS on ${BDF}"
        FAILED=$((FAILED + 1))
        continue
    fi

    NEW_VAL=$(setpci -v -s ${BDF} ECAP_ACS+0x6.w | awk '{print $NF}')
    if [ "${NEW_VAL}" != "0x0000" ]; then
        logger "Failed to disable ACS on ${BDF}"
        FAILED=$((FAILED + 1))
        continue
    fi

    CHANGED=$((CHANGED + 1))
done

echo "[INFO] ACS disabled on ${CHANGED} device(s); ${SKIPPED} had no ACS; ${FAILED} failed"
if [ "${FAILED}" -gt 0 ]; then
    echo "[WARN] Some devices could not be changed — check dmesg / syslog for details" >&2
    exit 1
fi
echo "[INFO] Reboot the node for the change to take full effect."
