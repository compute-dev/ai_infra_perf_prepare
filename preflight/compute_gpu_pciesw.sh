#!/bin/bash
# ============================================================================
# compute_gpu_pciesw.sh
#   Identify the PCIe Switch devices that connect NVIDIA GPUs on a server, and
#   output their key info:
#     - BDF address
#     - Vendor:Device ID / vendor name / model
#     - PCIe generation and link width (LnkCap / LnkSta)
#     - Upstream / Downstream role
#     - The list of GPUs hanging off each switch
#
# Usage:
#     bash compute_gpu_pciesw.sh                   # default summary table
#     bash compute_gpu_pciesw.sh -v                # verbose mode, prints the full topology chain
#     sudo bash compute_gpu_pciesw.sh              # root recommended for complete PCIe Cap fields
#
# Dependencies: nvidia-smi, lspci (pciutils), awk, sed
# ============================================================================

set -u

# ---------------------------- config ----------------------------
VERBOSE=0
while getopts "vh" opt; do
    case $opt in
        v) VERBOSE=1 ;;
        h)
            sed -n '2,14p' "$0"
            exit 0
            ;;
    esac
done

# ---------------------------- colors ----------------------------
if [ -t 1 ]; then
    C_RED='\033[0;31m'; C_GRN='\033[0;32m'; C_YLW='\033[0;33m'
    C_BLU='\033[0;34m'; C_CYN='\033[0;36m'; C_BLD='\033[1m'; C_RST='\033[0m'
else
    C_RED=''; C_GRN=''; C_YLW=''; C_BLU=''; C_CYN=''; C_BLD=''; C_RST=''
fi

log_info()  { echo -e "${C_CYN}[INFO]${C_RST}  $*"; }
log_warn()  { echo -e "${C_YLW}[WARN]${C_RST}  $*"; }
log_error() { echo -e "${C_RED}[ERROR]${C_RST} $*" >&2; }
log_head()  { echo -e "\n${C_BLD}${C_BLU}==== $* ====${C_RST}"; }

# ---------------------------- dependency check ----------------------------
for cmd in lspci awk sed; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log_error "missing dependency: $cmd"
        exit 1
    fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_warn "nvidia-smi not detected; will fall back to lspci for NVIDIA GPU detection"
    USE_NVIDIA_SMI=0
else
    USE_NVIDIA_SMI=1
fi

if [ "$(id -u)" -ne 0 ]; then
    log_warn "not running as root; some PCIe Capability fields may be unreadable, use sudo"
fi

# ---------------------------- helper functions ----------------------------

# Normalise an nvidia-smi BDF (e.g. 00000000:15:00.0) to lspci format (0000:15:00.0)
normalize_bdf() {
    local bdf="$1"
    # nvidia-smi may emit an 8-digit domain; lspci uses 4 digits
    bdf=$(echo "$bdf" | tr 'A-Z' 'a-z')
    if [[ "$bdf" =~ ^[0-9a-f]{8}: ]]; then
        bdf="${bdf:4}"
    fi
    # If there is no domain prefix, prepend 0000:
    if [[ ! "$bdf" =~ ^[0-9a-f]{4}: ]]; then
        bdf="0000:$bdf"
    fi
    echo "$bdf"
}

# Read a PCIe Bridge attribute from sysfs (parsed via lspci -vvv)
# Usage: pci_cap_type <BDF>  ->  Root Port / Upstream Port / Downstream Port / Endpoint / ""
pci_cap_type() {
    local bdf="$1"
    lspci -vvv -s "$bdf" 2>/dev/null \
        | awk -F'Express \\(v[0-9]+\\) ' '/Express \(v/ {
              sub(/,.*/, "", $2); print $2; exit
          }'
}

# Read LnkCap (max capability) and LnkSta (current state)
# Output: "SpeedGT/s xWidth"
pci_link_cap() {
    local bdf="$1"
    lspci -vvv -s "$bdf" 2>/dev/null \
        | awk '/LnkCap:/ {
              for (i = 1; i <= NF; i++) {
                  if ($i ~ /Speed/) speed = $(i+1);
                  if ($i ~ /Width/) width = $(i+1);
              }
              gsub(",", "", speed); gsub(",", "", width);
              print speed " " width; exit
          }'
}

pci_link_sta() {
    local bdf="$1"
    lspci -vvv -s "$bdf" 2>/dev/null \
        | awk '/LnkSta:/ {
              for (i = 1; i <= NF; i++) {
                  if ($i ~ /Speed/) speed = $(i+1);
                  if ($i ~ /Width/) width = $(i+1);
              }
              gsub(",", "", speed); gsub(",", "", width);
              print speed " " width; exit
          }'
}

# Translate a PCIe Speed (e.g. 32GT/s) to a Gen number
speed_to_gen() {
    case "$1" in
        2.5GT/s)  echo "Gen1" ;;
        5GT/s)    echo "Gen2" ;;
        8GT/s)    echo "Gen3" ;;
        16GT/s)   echo "Gen4" ;;
        32GT/s)   echo "Gen5" ;;
        64GT/s)   echo "Gen6" ;;
        *)        echo "Unknown($1)" ;;
    esac
}

# Get a BDF's Vendor:Device ID and human-readable name
pci_id_name() {
    local bdf="$1"
    lspci -nn -s "$bdf" 2>/dev/null | sed 's/^[^ ]* //'
}

# Walk sysfs upward to find the parent PCI device; return the parent BDF, or empty if at root
pci_parent() {
    local bdf="$1"
    local path
    path=$(readlink -f "/sys/bus/pci/devices/${bdf}" 2>/dev/null)
    [ -z "$path" ] && return
    local parent
    parent=$(dirname "$path")
    # If the parent path looks like .../0000:xx:xx.x there is a higher-level PCI device
    local pbdf
    pbdf=$(basename "$parent")
    if [[ "$pbdf" =~ ^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$ ]]; then
        echo "$pbdf"
    fi
}

# ---------------------------- collect GPU BDFs ----------------------------
log_head "Step 1. Collect NVIDIA GPU list"

declare -a GPU_BDFS=()
declare -A GPU_NAME=()

if [ "$USE_NVIDIA_SMI" -eq 1 ]; then
    while IFS=',' read -r idx name bdf; do
        idx=$(echo "$idx" | xargs)
        name=$(echo "$name" | xargs)
        bdf=$(normalize_bdf "$(echo "$bdf" | xargs)")
        GPU_BDFS+=("$bdf")
        GPU_NAME["$bdf"]="GPU${idx} ${name}"
        echo "  - GPU${idx}  ${bdf}  ${name}"
    done < <(nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv,noheader 2>/dev/null)
fi

# Fallback: if nvidia-smi is unavailable or returns nothing, use lspci for NVIDIA 3D controllers
if [ "${#GPU_BDFS[@]}" -eq 0 ]; then
    log_warn "nvidia-smi returned no GPU; falling back to lspci detection"
    while read -r line; do
        local_bdf=$(echo "$line" | awk '{print $1}')
        local_bdf=$(normalize_bdf "$local_bdf")
        GPU_BDFS+=("$local_bdf")
        GPU_NAME["$local_bdf"]=$(echo "$line" | cut -d' ' -f2-)
        echo "  - ${local_bdf}  $(echo "$line" | cut -d' ' -f2-)"
    done < <(lspci -D -nn | grep -Ei '3D controller.*NVIDIA|VGA.*NVIDIA')
fi

if [ "${#GPU_BDFS[@]}" -eq 0 ]; then
    log_error "no NVIDIA GPU found"
    exit 2
fi

log_info "found ${#GPU_BDFS[@]} GPU(s)"

# ---------------------------- walk the PCIe topology upward ----------------------------
log_head "Step 2. Walk the PCIe topology upward for each GPU"

# SWITCH_GPUS[<switch_bdf>] = "gpu_bdf1,gpu_bdf2,..."
declare -A SWITCH_GPUS=()
# SWITCH_ROLE[<switch_bdf>] = "Upstream Port" or "Downstream Port"
declare -A SWITCH_ROLE=()

for gpu in "${GPU_BDFS[@]}"; do
    [ "$VERBOSE" -eq 1 ] && echo -e "\n${C_GRN}v ${GPU_NAME[$gpu]} (${gpu}) upstream chain:${C_RST}"

    current="$gpu"
    hop=0
    while : ; do
        parent=$(pci_parent "$current")
        [ -z "$parent" ] && break   # reached root

        cap=$(pci_cap_type "$parent")
        id_name=$(pci_id_name "$parent")

        if [ "$VERBOSE" -eq 1 ]; then
            printf "   %s..- [%s] %-22s  %s\n" \
                "$(printf '  %.0s' $(seq 0 $hop))" \
                "$parent" \
                "${cap:-<no PCIe Cap>}" \
                "$id_name"
        fi

        # Root Port means we reached the CPU; stop walking up
        if [[ "$cap" == *"Root Port"* ]]; then
            break
        fi

        # Upstream / Downstream Port both belong to a third-party PCIe Switch component
        if [[ "$cap" == *"Upstream Port"* || "$cap" == *"Downstream Port"* ]]; then
            SWITCH_ROLE["$parent"]="$cap"
            if [ -z "${SWITCH_GPUS[$parent]:-}" ]; then
                SWITCH_GPUS["$parent"]="$gpu"
            else
                SWITCH_GPUS["$parent"]="${SWITCH_GPUS[$parent]},$gpu"
            fi
        fi

        current="$parent"
        hop=$((hop + 1))
        # Safety valve: prevent an anomalous loop
        [ "$hop" -gt 16 ] && break
    done
done

# ---------------------------- summary output ----------------------------
log_head "Step 3. PCIe Switch summary"

if [ "${#SWITCH_ROLE[@]}" -eq 0 ]; then
    log_warn "no third-party PCIe Switch found -- GPUs connect directly to CPU Root Ports"
    log_info "this is typical for HGX integrated systems (GPU direct-to-CPU) or NVLink-only topologies"
    exit 0
fi

# Header
printf "\n${C_BLD}%-14s %-20s %-18s %-10s %-12s %-12s %s${C_RST}\n" \
    "BDF" "Vendor:Device" "Role" "Max Gen" "Max Width" "Cur Gen" "Downstream GPUs"
printf -- '-%.0s' {1..130}; echo

# Output sorted by BDF
for sw in $(echo "${!SWITCH_ROLE[@]}" | tr ' ' '\n' | sort); do
    role="${SWITCH_ROLE[$sw]}"
    id_name=$(pci_id_name "$sw")
    # Extract [xxxx:yyyy]
    vid_did=$(echo "$id_name" | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{4}\]' | tr -d '[]' | head -n1)
    vendor_name=$(echo "$id_name" | sed -E 's/\[[0-9a-f]{4}:[0-9a-f]{4}\].*//' | awk -F': ' '{print $2}' | awk -F',' '{print $1}' | xargs)

    read -r cap_speed cap_width <<<"$(pci_link_cap "$sw")"
    read -r sta_speed _        <<<"$(pci_link_sta "$sw")"
    cap_gen=$(speed_to_gen "$cap_speed")
    sta_gen=$(speed_to_gen "$sta_speed")

    # Downstream GPUs: show only for Downstream Ports
    gpus="${SWITCH_GPUS[$sw]:-}"
    gpu_display=""
    if [ -n "$gpus" ]; then
        IFS=',' read -r -a arr <<<"$gpus"
        for g in "${arr[@]}"; do
            gpu_display+="${g##0000:} "
        done
    fi

    printf "%-14s %-20s %-18s %-10s %-12s %-12s %s\n" \
        "${sw##0000:}" \
        "${vid_did:-N/A}" \
        "${role}" \
        "${cap_gen}" \
        "${cap_width:-N/A}" \
        "${sta_gen}" \
        "${gpu_display:--}"
done

# ---------------------------- switch vendor lookup ----------------------------
log_head "Step 4. PCIe Switch vendor identification"

declare -A VENDOR_MAP=(
    ["1000"]="Broadcom/LSI (PLX)"
    ["10b5"]="Broadcom (PLX PEX series)"
    ["11f8"]="Microsemi/Microchip (Switchtec)"
    ["1b4b"]="Microsemi/Microchip"
    ["1dd8"]="Astera Labs"
    ["8086"]="Intel"
    ["1022"]="AMD"
    ["1a03"]="ASPEED (BMC, usually not a Switch)"
)

declare -A VENDOR_COUNT=()
for sw in "${!SWITCH_ROLE[@]}"; do
    vid=$(pci_id_name "$sw" | grep -oE '\[[0-9a-f]{4}:[0-9a-f]{4}\]' | tr -d '[]' | cut -d: -f1 | head -n1)
    [ -z "$vid" ] && continue
    VENDOR_COUNT["$vid"]=$(( ${VENDOR_COUNT[$vid]:-0} + 1 ))
done

for vid in "${!VENDOR_COUNT[@]}"; do
    vendor="${VENDOR_MAP[$vid]:-Unknown}"
    printf "  Vendor ${C_YLW}%s${C_RST}  (%s)  x  %d ports\n" \
        "$vid" "$vendor" "${VENDOR_COUNT[$vid]}"
done

# ---------------------------- conclusion ----------------------------
log_head "Step 5. Conclusion"

# Determine whether all switches are Gen5 (from LnkCap max rate)
all_gen5=1; any_gen=""
for sw in "${!SWITCH_ROLE[@]}"; do
    read -r cap_speed _ <<<"$(pci_link_cap "$sw")"
    g=$(speed_to_gen "$cap_speed")
    any_gen="$g"
    [ "$g" != "Gen5" ] && all_gen5=0
done

if [ "$all_gen5" -eq 1 ]; then
    echo -e "  ${C_GRN}OK: all PCIe Switch ports support Gen5 (32GT/s)${C_RST}"
else
    echo -e "  ${C_YLW}WARN: a non-Gen5 port is present; highest gen: ${any_gen}${C_RST}"
fi

unique_switches=$(for sw in "${!SWITCH_ROLE[@]}"; do
    # The Upstream and several Downstreams of the same physical Switch share a Bus number
    echo "${sw%:*}" | cut -d: -f2
done | sort -u | wc -l)

echo -e "  ${C_CYN}INFO: detected ${unique_switches} independent PCIe Switch chips (grouped by Bus number)${C_RST}"
echo -e "  ${C_CYN}INFO: ${#SWITCH_ROLE[@]} Switch ports total (Upstream + Downstream)${C_RST}"

exit 0
