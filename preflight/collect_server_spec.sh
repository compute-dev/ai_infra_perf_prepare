#!/bin/bash
# ============================================================================
# collect_server_spec.sh
#   Collect key server hardware specs for acceptance/audit, outputting an
#   "item / spec" summary aligned with the vendor's nameplate table.
#   Coverage:
#     - GPU         (model + count, e.g. GB200 x 4)
#     - CPU         (model + physical cores x sockets, e.g. NVIDIA Grace CPU 72 cores x 2)
#     - Memory      (per-DIMM / per-socket / total, type LPDDR5x/DDR5 etc.)
#     - Storage(OS) (capacity x count of the disk holding the root partition)
#     - Storage(Data)(data-disk capacity x count)
#     - NIC compute (ConnectX-7/8 compute NICs: measured link rate x port count + chip spec cap)
#     - NIC storage (BlueField-3 / DPU storage NICs, x count)
#
# Usage:
#     bash collect_server_spec.sh                 # standard summary
#     bash collect_server_spec.sh -v              # verbose mode (print raw command output)
#     bash collect_server_spec.sh -c              # print reproduction commands next to each step
#     sudo bash collect_server_spec.sh            # root recommended: dmidecode / nvme need it
#     bash collect_server_spec.sh -o spec.txt     # also save the summary to a file
#     bash collect_server_spec.sh -j              # single-line JSON output (parsed by the cluster checker)
#     bash collect_server_spec.sh -q              # quiet mode (only the final table to stdout)
#
# Dependencies: lspci, lscpu, awk, sed; optional: nvidia-smi, dmidecode, lsblk, ethtool, nvme
# ============================================================================

set -u

# ---------------------------- args ----------------------------
VERBOSE=0
OUT_FILE=""
JSON_MODE=0     # -j: JSON output (for the cluster consistency checker to parse)
QUIET=0         # -q: quiet mode (only final result; progress logs go to stderr)
SHOW_CMD=0      # -c: print reproduction commands next to each step
while getopts "vhjqco:" opt; do
    case $opt in
        v) VERBOSE=1 ;;
        o) OUT_FILE="$OPTARG" ;;
        j) JSON_MODE=1; QUIET=1 ;;
        q) QUIET=1 ;;
        c) SHOW_CMD=1 ;;
        h)
            sed -n '2,22p' "$0"
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

log_info()  { [ "$QUIET" -eq 1 ] && echo -e "${C_CYN}[INFO]${C_RST}  $*" >&2 || echo -e "${C_CYN}[INFO]${C_RST}  $*"; }
log_warn()  { [ "$QUIET" -eq 1 ] && echo -e "${C_YLW}[WARN]${C_RST}  $*" >&2 || echo -e "${C_YLW}[WARN]${C_RST}  $*"; }
log_error() { echo -e "${C_RED}[ERROR]${C_RST} $*" >&2; }
log_head()  { [ "$QUIET" -eq 1 ] && echo -e "\n${C_BLD}${C_BLU}==== $* ====${C_RST}" >&2 || echo -e "\n${C_BLD}${C_BLU}==== $* ====${C_RST}"; }
# In quiet mode, ordinary progress info also goes to stderr; stdout keeps only the final result.
qecho() { [ "$QUIET" -eq 1 ] && echo -e "$*" >&2 || echo -e "$*"; }

vlog() { [ "$VERBOSE" -eq 1 ] && echo -e "${C_YLW}[verbose]${C_RST} $*"; }

# Print the raw command used to reproduce a step (active when -c is on).
# Usage: log_cmd "cmd1" "cmd2" ...
log_cmd() {
    [ "$SHOW_CMD" -ne 1 ] && return
    local prefix="${C_GRN}[cmd]${C_RST}"
    local target
    if [ "$QUIET" -eq 1 ]; then target=2; else target=1; fi
    for c in "$@"; do
        if [ "$target" -eq 2 ]; then
            echo -e "  ${prefix} $ ${c}" >&2
        else
            echo -e "  ${prefix} $ ${c}"
        fi
    done
}

# ---------------------------- dependency check ----------------------------
need_root=0
[ "$(id -u)" -ne 0 ] && need_root=1

has() { command -v "$1" >/dev/null 2>&1; }

for cmd in lspci awk sed lscpu; do
    if ! has "$cmd"; then
        log_error "missing required tool: $cmd"
        exit 1
    fi
done

[ "$need_root" -eq 1 ] && log_warn "not running as root: dmidecode / nvme / ethtool fields may be incomplete, use sudo"

# ---------------------------- helper functions ----------------------------

# Convert a byte count to GiB / TiB (1 decimal place).
human_size() {
    local bytes="$1"
    awk -v b="$bytes" 'BEGIN{
        if (b == "" || b+0 == 0) { print "N/A"; exit }
        if (b >= 1099511627776) printf "%.1fT\n", b/1099511627776;
        else if (b >= 1073741824) printf "%.1fG\n", b/1073741824;
        else if (b >= 1048576)    printf "%.1fM\n", b/1048576;
        else printf "%dB\n", b;
    }'
}

# ---------------------------- Step 1. GPU ----------------------------
log_head "Step 1. GPU"
log_cmd "nvidia-smi --query-gpu=name --format=csv,noheader" \
        "# or (no driver) fallback: lspci -nn | grep -Ei '3D controller.*NVIDIA|VGA.*NVIDIA'"

GPU_MODEL="N/A"
GPU_COUNT=0
GPU_DETAIL=""

if has nvidia-smi; then
    # Take all GPU names, dedupe & count
    mapfile -t _gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
    GPU_COUNT=${#_gpu_names[@]}
    if [ "$GPU_COUNT" -gt 0 ]; then
        # Most frequent model as the representative
        GPU_MODEL=$(printf '%s\n' "${_gpu_names[@]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sort | uniq -c | sort -rn | head -n1 | awk '{$1=""; sub(/^ /,""); print}')
        GPU_DETAIL=$(printf '%s\n' "${_gpu_names[@]}" | sort | uniq -c | awk '{cnt=$1; $1=""; sub(/^ /,""); printf "    - %s x %d\n", $0, cnt}')
    fi
else
    # Fallback: lspci for NVIDIA 3D controller / VGA
    mapfile -t _gpu_names < <(lspci -nn 2>/dev/null | grep -Ei '3D controller.*NVIDIA|VGA.*NVIDIA' | sed -E 's/^[^ ]+ [^:]+: //; s/ \[[0-9a-f]{4}:[0-9a-f]{4}\].*//')
    GPU_COUNT=${#_gpu_names[@]}
    if [ "$GPU_COUNT" -gt 0 ]; then
        GPU_MODEL=$(printf '%s\n' "${_gpu_names[@]}" | sort | uniq -c | sort -rn | head -n1 | awk '{$1=""; sub(/^ /,""); print}')
        GPU_DETAIL=$(printf '%s\n' "${_gpu_names[@]}" | sort | uniq -c | awk '{cnt=$1; $1=""; sub(/^ /,""); printf "    - %s x %d\n", $0, cnt}')
    fi
fi

if [ "$GPU_COUNT" -eq 0 ]; then
    log_warn "no GPU detected"
    GPU_SUMMARY="N/A"
else
    GPU_SUMMARY="${GPU_MODEL} x ${GPU_COUNT}"
    log_info "GPU: ${GPU_SUMMARY}"
    [ -n "$GPU_DETAIL" ] && qecho "${GPU_DETAIL}"
fi

# ---------------------------- Step 2. CPU ----------------------------
log_head "Step 2. CPU"
log_cmd "lscpu | grep -E 'Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|^CPU\(s\)'"

CPU_MODEL=$(lscpu | awk -F: '/Model name/ {sub(/^ +/, "", $2); print $2; exit}')
CPU_SOCKETS=$(lscpu | awk -F: '/Socket\(s\)/ {gsub(/ /, "", $2); print $2; exit}')
CORES_PER_SOCKET=$(lscpu | awk -F: '/Core\(s\) per socket/ {gsub(/ /, "", $2); print $2; exit}')
THREADS_PER_CORE=$(lscpu | awk -F: '/Thread\(s\) per core/ {gsub(/ /, "", $2); print $2; exit}')
CPU_TOTAL_LOGICAL=$(lscpu | awk -F: '/^CPU\(s\)/ {gsub(/ /, "", $2); print $2; exit}')

[ -z "$CPU_MODEL" ] && CPU_MODEL="Unknown"
[ -z "$CPU_SOCKETS" ] && CPU_SOCKETS=1
[ -z "$CORES_PER_SOCKET" ] && CORES_PER_SOCKET=0

CPU_SUMMARY="${CPU_MODEL} (${CORES_PER_SOCKET} cores x ${CPU_SOCKETS} per Tray)"
log_info "CPU: ${CPU_SUMMARY}"
qecho "    - Sockets             : ${CPU_SOCKETS}"
qecho "    - Cores per socket    : ${CORES_PER_SOCKET}"
qecho "    - Threads per core    : ${THREADS_PER_CORE}"
qecho "    - Total logical CPUs  : ${CPU_TOTAL_LOGICAL}"
[ "$VERBOSE" -eq 1 ] && lscpu | sed 's/^/      /' >&2

# ---------------------------- Step 3. Memory ----------------------------
log_head "Step 3. Memory"
log_cmd "awk '/MemTotal/ {print \$2}' /proc/meminfo    # total (KB)" \
        "sudo dmidecode -t memory | grep -E 'Type:|Size:'    # memory type & per-DIMM size"

MEM_TOTAL_KB=$(awk '/MemTotal/ {print $2; exit}' /proc/meminfo)
MEM_TOTAL_BYTES=$(( MEM_TOTAL_KB * 1024 ))
MEM_TOTAL_HUMAN=$(human_size "$MEM_TOTAL_BYTES")

MEM_TYPE="Unknown"
MEM_PER_DIMM=""
MEM_DIMM_COUNT=0
MEM_PER_SOCKET="N/A"

if has dmidecode && [ "$need_root" -eq 0 ]; then
    # Collect all installed DIMMs
    DMIDECODE_OUT=$(dmidecode -t memory 2>/dev/null || true)
    if [ -n "$DMIDECODE_OUT" ]; then
        # Type (DDR4 / DDR5 / LPDDR5 / LPDDR5x ...)
        MEM_TYPE=$(echo "$DMIDECODE_OUT" | awk -F: '/^[[:space:]]*Type:/ {gsub(/ /, "", $2); if ($2 != "Unknown" && $2 != "Other" && $2 != "") {print $2; exit}}')
        [ -z "$MEM_TYPE" ] && MEM_TYPE="Unknown"

        # Per-DIMM size + count (installed DIMMs only)
        MEM_DIMM_INFO=$(echo "$DMIDECODE_OUT" | awk '
            /^Memory Device/ {in_dev=1; size=""; next}
            in_dev && /^[[:space:]]*Size:/ {
                sub(/^[[:space:]]*Size:[[:space:]]*/, "");
                size=$0
            }
            /^$/ && in_dev {
                if (size != "" && size !~ /No Module Installed|Not Installed/) print size;
                in_dev=0; size=""
            }')
        MEM_DIMM_COUNT=$(echo "$MEM_DIMM_INFO" | grep -c .)
        MEM_PER_DIMM=$(echo "$MEM_DIMM_INFO" | sort | uniq -c | sort -rn | head -n1 | awk '{$1=""; sub(/^ /,""); print}')
    fi
else
    log_warn "dmidecode not run (needs root); memory type / per-DIMM size unavailable"
fi

# Per-socket capacity = total / sockets
if [ "$CPU_SOCKETS" -gt 0 ] 2>/dev/null; then
    MEM_PER_SOCKET=$(awk -v t="$MEM_TOTAL_BYTES" -v s="$CPU_SOCKETS" 'BEGIN{
        v=t/s;
        if (v >= 1099511627776) printf "%.0fT\n", v/1099511627776;
        else if (v >= 1073741824) printf "%.0fG\n", v/1073741824;
        else printf "%.0fM\n", v/1048576;
    }')
fi

MEM_SUMMARY="${MEM_PER_SOCKET} ${MEM_TYPE} per CPU (${MEM_TOTAL_HUMAN} per Tray)"
log_info "Memory: ${MEM_SUMMARY}"
qecho "    - Total              : ${MEM_TOTAL_HUMAN}"
qecho "    - Type               : ${MEM_TYPE}"
qecho "    - Per socket         : ${MEM_PER_SOCKET}"
[ -n "$MEM_PER_DIMM" ] && qecho "    - DIMM size x count  : ${MEM_PER_DIMM} x ${MEM_DIMM_COUNT}"

# ---------------------------- Step 4. Storage ----------------------------
log_head "Step 4. Storage (OS / Data)"
log_cmd "findmnt -no SOURCE /                          # find the device backing the root partition" \
        "lsblk -no PKNAME <root_partition>              # look up its parent physical disk" \
        "lsblk -dn -o NAME,SIZE,TYPE,MODEL              # enumerate all physical disks"

# Find the physical disk backing the root partition
ROOT_SRC=$(findmnt -no SOURCE / 2>/dev/null || df / | awk 'NR==2{print $1}')
ROOT_DISK=""
if [ -n "$ROOT_SRC" ] && has lsblk; then
    ROOT_DISK=$(lsblk -no PKNAME "$ROOT_SRC" 2>/dev/null | head -n1)
    [ -z "$ROOT_DISK" ] && ROOT_DISK=$(basename "$ROOT_SRC" | sed 's/[0-9]*$//')
fi

declare -A OS_DISKS=()
declare -A DATA_DISKS=()

if has lsblk; then
    # All TYPE=disk physical disks; filter out loop / rom
    while read -r name size type model; do
        [ "$type" != "disk" ] && continue
        # e.g. nvme0n1 / sda
        size_h="$size"
        # Classify into OS / Data
        if [ "$name" = "$ROOT_DISK" ]; then
            key="${size_h}|${model:-N/A}"
            OS_DISKS["$key"]=$(( ${OS_DISKS["$key"]:-0} + 1 ))
        else
            key="${size_h}|${model:-N/A}"
            DATA_DISKS["$key"]=$(( ${DATA_DISKS["$key"]:-0} + 1 ))
        fi
    done < <(lsblk -dn -o NAME,SIZE,TYPE,MODEL 2>/dev/null)
else
    log_warn "lsblk missing, skipping disk enumeration"
fi

# Output OS disk
OS_SUMMARY=""
qecho "  [OS Disk] (root on /dev/${ROOT_DISK:-?})"
if [ "${#OS_DISKS[@]}" -eq 0 ]; then
    qecho "    - unrecognized"
    OS_SUMMARY="N/A"
else
    for k in "${!OS_DISKS[@]}"; do
        sz="${k%%|*}"; md="${k##*|}"
        cnt="${OS_DISKS[$k]}"
        qecho "    - ${sz} x ${cnt}  (${md})"
        OS_SUMMARY+="${sz} x ${cnt} per Tray; "
    done
    OS_SUMMARY="${OS_SUMMARY%; }"
fi

# Output data disk (grouped by capacity + model)
DATA_SUMMARY=""
qecho "  [Data Disk]"
if [ "${#DATA_DISKS[@]}" -eq 0 ]; then
    qecho "    - unrecognized (or only an OS disk)"
    DATA_SUMMARY="N/A"
else
    for k in "${!DATA_DISKS[@]}"; do
        sz="${k%%|*}"; md="${k##*|}"
        cnt="${DATA_DISKS[$k]}"
        qecho "    - ${sz} x ${cnt}  (${md})"
        DATA_SUMMARY+="${sz} x ${cnt} per Tray; "
    done
    DATA_SUMMARY="${DATA_SUMMARY%; }"
fi

# ---------------------------- Step 5. NIC ----------------------------
log_head "Step 5. NIC (Compute / Storage)"
log_cmd "lspci -nn | grep -Ei 'Mellanox|\\[15b3:' | grep -Ei 'Ethernet|Infiniband|Network'   # enumerate Mellanox/NVIDIA NICs" \
        "ls /sys/bus/pci/devices/0000:<BDF>/net/                    # find net interfaces under this NIC" \
        "cat /sys/class/net/<iface>/operstate                       # port state up/down" \
        "cat /sys/class/net/<iface>/speed                           # negotiated rate (Mbps), /1000 -> Gb/s" \
        "# cross-check: ethtool <iface> | grep -E 'Speed|Link detected'" \
        "# IB mode: cat /sys/bus/pci/devices/0000:<BDF>/infiniband/*/ports/*/{state,rate}"

# Mellanox / NVIDIA NIC PCI vendor 0x15b3
declare -A NIC_COMPUTE=()   # key=<desc> | <chip_cap> | <ports_info>, val=count
declare -A NIC_STORAGE=()   # key=<desc> | <ports_info>,               val=count

# Chip-family spec cap (**reference only**, not this specific card's capability):
#   A given ASIC ships in multiple SKUs:
#     - ConnectX-6 Lx : 10/25/40/50 GbE     (single-port spec cap 50GbE)
#     - ConnectX-6 Dx : 25/50/100/200 GbE   (single-port spec cap 200GbE)
#     - ConnectX-6    : 100/200 GbE         (single-port spec cap 200GbE / HDR IB)
#     - ConnectX-7    : 200/400 GbE         (single-port spec cap 400GbE / NDR IB)
#     - ConnectX-8    : 400/800 GbE         (single-port spec cap 800GbE / XDR IB)
# lspci cannot distinguish SKUs, so this returns the "chip-family spec cap" for POC
# reference. **What really represents this card's actual capability is the negotiated
# link rate read from sysfs by probe_nic_ports() (needs cabling and link up).**
guess_chip_cap() {
    local name="$1"
    case "$name" in
        *ConnectX-8*|*CX8*)                echo "800GbE cap"; return ;;
        *"ConnectX-7"*|*CX7*)              echo "400GbE cap"; return ;;
        *"ConnectX-6 Lx"*|*CX6Lx*)         echo "50GbE cap";  return ;;
        *"ConnectX-6 Dx"*|*CX6Dx*)         echo "200GbE cap"; return ;;
        *ConnectX-6*|*CX6*)                echo "200GbE cap"; return ;;
        *ConnectX-5*|*CX5*)                echo "100GbE cap"; return ;;
    esac
    case "$name" in
        *MT2910*)                          echo "400GbE cap"; return ;;
        *MT2892*)                          echo "200GbE cap"; return ;;
        *MT2894*)                          echo "50GbE cap";  return ;;
        *MT28908*|*MT28800*)               echo "100GbE cap"; return ;;
    esac
    echo ""
}

# Port probe: from a PCI BDF, find all net interfaces under this card, count active
# ports and per-port link rate.
# Input: full BDF (e.g. 0000:81:00.0 or short 81:00.0)
# Output like:
#   "25Gb/s x 2 ports (up)"      both ports wired, each 25Gb/s
#   "50Gb/s x 1 port (up)"       only 1 port wired
#   "0 x 2 ports (link down)"    card present but neither up
#   "no netdev"                   driver not loaded or not in RoCE/Ethernet mode
probe_nic_ports() {
    local bdf="$1"
    # Normalise to a full BDF
    [[ "$bdf" != *:*:* ]] && bdf="0000:$bdf"

    local sys_dev="/sys/bus/pci/devices/${bdf}"
    [ ! -d "$sys_dev" ] && { echo "no sysfs"; return; }

    # Find all net interfaces under this PCI device
    local netdir="${sys_dev}/net"
    if [ ! -d "$netdir" ]; then
        # InfiniBand mode has no net/; fall back to an IB port check
        if [ -d "${sys_dev}/infiniband" ]; then
            local ib_ports=0 ib_active=0 ib_rate=""
            for p in "${sys_dev}"/infiniband/*/ports/*; do
                [ -d "$p" ] || continue
                ib_ports=$((ib_ports+1))
                local st rate
                st=$(cat "$p/state" 2>/dev/null | awk '{print $NF}')
                rate=$(cat "$p/rate" 2>/dev/null | awk '{print $1$2}')
                [ "$st" = "ACTIVE" ] && { ib_active=$((ib_active+1)); ib_rate="$rate"; }
            done
            if [ "$ib_active" -gt 0 ]; then
                echo "${ib_rate:-?} x ${ib_active} ports (IB active)"
            else
                echo "0 x ${ib_ports} ports (IB down)"
            fi
            return
        fi
        echo "no netdev"; return
    fi

    local total=0 up=0
    local -A speed_count=()
    local up_speed_list=""
    for nif in "$netdir"/*; do
        [ -d "$nif" ] || continue
        total=$((total+1))
        local ifname state spd
        ifname=$(basename "$nif")
        state=$(cat "$nif/operstate" 2>/dev/null)
        spd=$(cat "$nif/speed" 2>/dev/null)
        # speed may be -1 or read-error when down
        if [ "$state" = "up" ] && [ -n "$spd" ] && [ "$spd" -gt 0 ] 2>/dev/null; then
            up=$((up+1))
            # to Gb/s
            local g
            g=$(awk -v s="$spd" 'BEGIN{printf "%g", s/1000}')
            speed_count["${g}Gb/s"]=$(( ${speed_count["${g}Gb/s"]:-0} + 1 ))
            up_speed_list+="${ifname}=${g}Gb/s "
        fi
    done

    # In verbose, expose per-port detail
    [ "$VERBOSE" -eq 1 ] && [ -n "$up_speed_list" ] && \
        vlog "  ${bdf}: ${up_speed_list}" >&2

    if [ "$up" -eq 0 ]; then
        echo "0 x ${total} ports (link down)"
        return
    fi

    # Group by rate (normally both ports of a card share a rate; list all if not)
    if [ "${#speed_count[@]}" -eq 1 ]; then
        local sp; sp=$(echo "${!speed_count[@]}")
        local cnt; cnt=${speed_count["$sp"]}
        local plural="ports"; [ "$cnt" -eq 1 ] && plural="port"
        echo "${sp} x ${cnt} ${plural} (up)"
    else
        local out=""
        for sp in "${!speed_count[@]}"; do
            [ -n "$out" ] && out+=" + "
            out+="${speed_count[$sp]}x${sp}"
        done
        echo "${out} (up)"
    fi
}

# List all Mellanox/NVIDIA network-related devices (keep BDF)
NIC_LINES=$(lspci -nn 2>/dev/null | grep -Ei 'Mellanox|\[15b3:' | grep -Ei 'Ethernet|Infiniband|Network' || true)

while IFS= read -r line; do
    [ -z "$line" ] && continue
    # First field is the BDF (e.g. 81:00.0)
    bdf=$(echo "$line" | awk '{print $1}')
    # Extract human-readable name (drop BDF and vendor:device id)
    desc=$(echo "$line" | sed -E 's/^[^ ]+ //; s/ \[[0-9a-f]{4}:[0-9a-f]{4}\]( \(rev [0-9a-f]+\))?$//')
    chip_cap=$(guess_chip_cap "$desc")
    ports_info=$(probe_nic_ports "$bdf")

    if echo "$desc" | grep -qiE 'BlueField|DPU'; then
        # storage / DPU NIC (key = "<desc> | <ports_info>")
        key="${desc} | ${ports_info}"
        NIC_STORAGE["$key"]=$(( ${NIC_STORAGE["$key"]:-0} + 1 ))
    else
        # compute NIC (key = "<desc> | <chip_cap> | <ports_info>")
        key="${desc} | ${chip_cap:-?} | ${ports_info}"
        NIC_COMPUTE["$key"]=$(( ${NIC_COMPUTE["$key"]:-0} + 1 ))
    fi
done <<< "$NIC_LINES"

# Output compute NIC
NIC_COMPUTE_SUMMARY=""
qecho "  [NIC Compute]"
if [ "${#NIC_COMPUTE[@]}" -eq 0 ]; then
    qecho "    - no Mellanox/NVIDIA compute NIC found"
    NIC_COMPUTE_SUMMARY="N/A"
else
    for k in "${!NIC_COMPUTE[@]}"; do
        cnt="${NIC_COMPUTE[$k]}"
        # key = "<desc> | <chip_cap> | <ports_info>"  split on " | " into 3 parts
        desc="${k%% | *}"
        rest="${k#* | }"
        chip_cap="${rest%% | *}"
        ports="${rest#* | }"
        # short model: ConnectX-7 / ConnectX-8 / ConnectX-6 Lx ...
        short=$(echo "$desc" | grep -oE 'ConnectX-[0-9]+( Lx| Dx)?' | head -n1)
        [ -z "$short" ] && short="$desc"
        qecho "    - ${desc}  x ${cnt}   [link: ${ports}]  (chip cap: ${chip_cap})"
        if [ -n "$NIC_COMPUTE_SUMMARY" ]; then NIC_COMPUTE_SUMMARY+="; "; fi
        # e.g. ConnectX-7 (link 400Gb/s x 2 ports (up); chip cap 400GbE cap) x 4 per Tray
        NIC_COMPUTE_SUMMARY+="${short} (link ${ports}; chip cap ${chip_cap}) x ${cnt} per Tray"
    done
fi

# Output storage NIC (BlueField)
NIC_STORAGE_SUMMARY=""
qecho "  [NIC Storage / DPU]"
if [ "${#NIC_STORAGE[@]}" -eq 0 ]; then
    qecho "    - no BlueField/DPU storage NIC found"
    NIC_STORAGE_SUMMARY="N/A"
else
    for k in "${!NIC_STORAGE[@]}"; do
        cnt="${NIC_STORAGE[$k]}"
        # key = "<desc> | <ports_info>"
        desc="${k%% | *}"
        ports="${k#* | }"
        short=$(echo "$desc" | grep -oE 'BlueField-[0-9]+' | head -n1)
        [ -z "$short" ] && short="BlueField"
        # BlueField-3 contains ConnectX-7; BlueField-2 contains ConnectX-6 Dx
        inner=""
        case "$short" in
            BlueField-3) inner=" (ConnectX-7)" ;;
            BlueField-2) inner=" (ConnectX-6 Dx)" ;;
        esac
        qecho "    - ${desc}  x ${cnt}   [link: ${ports}]"
        if [ -n "$NIC_STORAGE_SUMMARY" ]; then NIC_STORAGE_SUMMARY+="; "; fi
        NIC_STORAGE_SUMMARY+="${short}${inner} x ${cnt} per Tray (link ${ports})"
    done
fi

[ "$VERBOSE" -eq 1 ] && {
    vlog "raw NIC list:"
    echo "$NIC_LINES" | sed 's/^/      /' >&2
}

# ---------------------------- Step 6. summary output ----------------------------
log_head "Step 6. Server spec summary (for acceptance/audit)"

# Build the final table
TABLE=$(cat <<EOF
Item              Spec
GPU               ${GPU_SUMMARY}
CPU               ${CPU_SUMMARY}
Memory            ${MEM_SUMMARY}
Storage (OS)      ${OS_SUMMARY}
Storage (Data)    ${DATA_SUMMARY}
NIC compute       ${NIC_COMPUTE_SUMMARY}
NIC storage       ${NIC_STORAGE_SUMMARY}
EOF
)

# JSON mode: stdout outputs only single-line JSON, for the cluster checker to parse
if [ "$JSON_MODE" -eq 1 ]; then
    # Minimal JSON escaping (double quotes, backslash) via sed
    j_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
    cat <<EOF
{"hostname":"$(j_escape "$(hostname)")","gpu":"$(j_escape "$GPU_SUMMARY")","gpu_count":${GPU_COUNT:-0},"gpu_model":"$(j_escape "$GPU_MODEL")","cpu":"$(j_escape "$CPU_SUMMARY")","cpu_model":"$(j_escape "$CPU_MODEL")","cpu_sockets":${CPU_SOCKETS:-0},"cores_per_socket":${CORES_PER_SOCKET:-0},"memory":"$(j_escape "$MEM_SUMMARY")","mem_total":"$(j_escape "$MEM_TOTAL_HUMAN")","mem_type":"$(j_escape "$MEM_TYPE")","storage_os":"$(j_escape "$OS_SUMMARY")","storage_data":"$(j_escape "$DATA_SUMMARY")","nic_compute":"$(j_escape "$NIC_COMPUTE_SUMMARY")","nic_storage":"$(j_escape "$NIC_STORAGE_SUMMARY")"}
EOF
else
    # Terminal-highlighted version (column-sep on ">=2 spaces" to avoid splitting labels with spaces like "Storage (OS)" / "NIC compute")
    echo -e "${C_BLD}"
    echo "$TABLE" | awk -F'  +' 'NR==1{printf "%-18s%s\n", $1, $2; next} {printf "%-18s%s\n", $1, $2}'
    echo -e "${C_RST}"
fi

# Write to file (plain text, no color)
if [ -n "$OUT_FILE" ]; then
    {
        echo "# Server Spec Report  ($(date '+%F %T'))"
        echo "# Hostname: $(hostname)"
        echo
        echo "$TABLE" | awk -F'  +' 'NR==1{printf "%-18s%s\n", $1, $2; next} {printf "%-18s%s\n", $1, $2}'
    } > "$OUT_FILE"
    log_info "result written to: ${OUT_FILE}"
fi

exit 0
