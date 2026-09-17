#!/bin/bash
# ============================================================================
# collect_runtime_state.sh
#   Collect a server's "software / runtime-state" key items -- the complement of
#   collect_server_spec.sh (hardware spec). These items are invisible when
#   nvidia-smi's driver_version matches across nodes, yet they directly decide
#   whether cuBLAS / single-node / multi-node / bitwise tests can be compared
#   apples-to-apples.
#
#   Coverage (the test-sensitivity each item introduces):
#     B1 driver / CUDA-API / userspace libcuda / cuda-compat  (bitwise + perf critical)
#     B2 ECC mode + cumulative uncorrected errors             (marginal / degraded cards)
#     B3 GPU clock cap / power cap / throttling               (cuBLAS weak-card baseline)
#     B4 each compute HCA present + ACTIVE + rate + firmware   (multi-node link-up)
#     B5 OFED / fabricmanager / fabric state / peermem         (multi-node & RDMA)
#     B6 kernel / OS / persistence / memlock / NUMA            (debug + bitwise record)
#     B7 container bench image id                              (TFLOPs comparability)
#     B8 Fabric / RoCE QoS / GPU-NIC topology / NVLink         (pre-NCCL-test must-check)
#
#   Multi-GPU / multi-HCA values on one host are collapsed into a single
#   cross-host-comparable scalar:
#     all-identical -> that value; empty -> N/A; differing -> "mixed:v1|v2|.."
#     (cluster_runtime_check reports a diff on that).
#
# Usage:
#     bash collect_runtime_state.sh          # human-readable table
#     bash collect_runtime_state.sh -c       # also print the reproduction command per item
#     bash collect_runtime_state.sh -j       # single-line JSON (parsed by cluster_runtime_check)
#     bash collect_runtime_state.sh -o rt.txt
#     sudo bash collect_runtime_state.sh     # root recommended: dmidecode/nvme not needed, but fuller
#
# Dependencies: awk sed; optional: nvidia-smi ibstat ofed_info lsmod docker
#
# Environment variables:
#     COMPUTE_HCA_RE  When set, used as an anchored ERE to select compute HCAs by name
#                     (e.g. COMPUTE_HCA_RE='mlx5_10[1-8]'). When UNSET, the script
#                     auto-detects compute HCAs (see B4 below for the storage-NIC trap).
#     EXCLUDE_HCA     Comma-separated exact HCA names to exclude from the auto-detected
#                     set (e.g. EXCLUDE_HCA=mlx5_1,mlx5_6). Same convention as
#                     nccl_bench/gpu_nic_topo.sh. Only consulted when COMPUTE_HCA_RE is unset.
#     BENCH_IMAGE_RE  When set, grep this pattern to identify the bench container image.
#                     When unset, the bench_image_id field is tagged "not-scoped" rather
#                     than matching any product/registry name.
# ============================================================================
set -u

JSON_MODE=0; SHOW_CMD=0; OUT_FILE=""
while getopts "jco:h" opt; do
    case $opt in
        j) JSON_MODE=1 ;;
        c) SHOW_CMD=1 ;;
        o) OUT_FILE="$OPTARG" ;;
        h) sed -n '2,44p' "$0"; exit 0 ;;
    esac
done

if [ -t 1 ] && [ "$JSON_MODE" -eq 0 ]; then
    C_GRN='\033[0;32m'; C_YLW='\033[0;33m'; C_BLU='\033[0;34m'; C_CYN='\033[0;36m'
    C_BLD='\033[1m'; C_RST='\033[0m'
else
    C_GRN=''; C_YLW=''; C_BLU=''; C_CYN=''; C_BLD=''; C_RST=''
fi
# In JSON mode all progress logging goes to stderr; stdout keeps only the final JSON line.
say()  { [ "$JSON_MODE" -eq 1 ] && echo -e "$*" >&2 || echo -e "$*"; }
head_() { say "\n${C_BLD}${C_BLU}==== $* ====${C_RST}"; }
cmd_() { [ "$SHOW_CMD" -eq 1 ] && say "  ${C_GRN}[cmd]${C_RST} $ $*"; }
has()  { command -v "$1" >/dev/null 2>&1; }

# Collapse a multi-line value into a cross-host-comparable scalar:
#   all-identical -> that value; empty -> N/A; differing -> mixed:a|b
# Double quotes (which would break JSON) are replaced with '@'; commas are kept
# (the parser takes values by " boundaries).
collapse() {
    awk '
        { gsub(/^[ \t]+|[ \t]+$/, ""); if ($0=="") next; gsub(/"/,"@"); seen[$0]=1; order[++n]=$0 }
        END{
            c=0; for (k in seen) c++;
            if (c==0){ print "N/A"; exit }
            if (c==1){ for (k in seen){ print k } exit }
            # differing: dedupe and join in first-seen order
            out=""; delete done;
            for(i=1;i<=n;i++){ v=order[i]; if(!(v in done)){ done[v]=1; out=(out==""?v:out"|"v) } }
            print "mixed:" out
        }'
}

# ---- collection --------------------------------------------------------------
GPU_OK=0; has nvidia-smi && GPU_OK=1

# B1 driver / CUDA API / libcuda / cuda-compat
head_ "B1. Driver / CUDA / libcuda"
cmd_ "nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u"
cmd_ "nvidia-smi | awk -F'CUDA Version:' '/CUDA Version/{print \$2}'"
cmd_ "ldconfig -p | grep 'libcuda.so' ; readlink -f <libcuda.so.1>"
cmd_ "dpkg -l | grep cuda-compat  # or rpm -qa | grep cuda-compat"
if [ "$GPU_OK" -eq 1 ]; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | collapse)
    CUDA_API=$(nvidia-smi 2>/dev/null | sed -nE 's/.*CUDA Version: *([0-9.]+).*/\1/p' | head -1)
else
    DRIVER="N/A"; CUDA_API="N/A"
fi
[ -z "$CUDA_API" ] && CUDA_API="N/A"
# Userspace libcuda real resolved path & version (catch cuda-compat preemption)
LIBCUDA_RAW=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')
[ -z "$LIBCUDA_RAW" ] && LIBCUDA_RAW=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so/{print $NF; exit}')
LIBCUDA_REAL=$(readlink -f "$LIBCUDA_RAW" 2>/dev/null)
LIBCUDA_VER=$(basename "$LIBCUDA_REAL" 2>/dev/null | sed -E 's/^libcuda\.so\.?//')
[ -z "$LIBCUDA_VER" ] && LIBCUDA_VER="N/A"
# Whether cuda-compat is installed (the usual culprit)
if has dpkg; then
    CUDA_COMPAT=$(dpkg -l 2>/dev/null | awk '/cuda-compat/{print $2"("$3")"}' | paste -sd, -)
elif has rpm; then
    CUDA_COMPAT=$(rpm -qa 2>/dev/null | grep -i cuda-compat | paste -sd, -)
fi
[ -z "${CUDA_COMPAT:-}" ] && CUDA_COMPAT="none"
say "  driver_version : ${DRIVER}"
say "  cuda_api_max   : ${CUDA_API}"
say "  libcuda        : ${LIBCUDA_VER}  (${LIBCUDA_REAL:-N/A})"
say "  cuda_compat    : ${CUDA_COMPAT}"

# B2 ECC
head_ "B2. ECC"
cmd_ "nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader | sort -u"
cmd_ "nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader"
if [ "$GPU_OK" -eq 1 ]; then
    ECC_MODE=$(nvidia-smi --query-gpu=ecc.mode.current --format=csv,noheader 2>/dev/null | collapse)
    ECC_UNCORR=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader 2>/dev/null \
                 | awk '{s+=$1} END{print (NR?s:"N/A")}')
else
    ECC_MODE="N/A"; ECC_UNCORR="N/A"
fi
[ -z "$ECC_UNCORR" ] && ECC_UNCORR="N/A"
say "  ecc_mode       : ${ECC_MODE}"
say "  ecc_uncorr_sum : ${ECC_UNCORR}"

# B3 clock / power / throttling
head_ "B3. Clock / Power"
cmd_ "nvidia-smi --query-gpu=clocks.max.sm,power.limit,power.max_limit --format=csv,noheader"
cmd_ "nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader"
if [ "$GPU_OK" -eq 1 ]; then
    SM_MAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | collapse)
    PWR_LIMIT=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | collapse)
    PWR_MAX=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits 2>/dev/null | collapse)
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
else
    SM_MAX="N/A"; PWR_LIMIT="N/A"; PWR_MAX="N/A"; GPU_COUNT=0
fi
say "  gpu_count      : ${GPU_COUNT}"
say "  sm_clock_max   : ${SM_MAX} MHz"
say "  power_limit    : ${PWR_LIMIT} W (cap ${PWR_MAX} W)"

# B4 compute HCA presence / ACTIVE / rate / firmware
#   Compute-NIC naming is vendor-specific. When COMPUTE_HCA_RE is set, it is used as
#   an anchored ERE to match device names (original behaviour). When COMPUTE_HCA_RE is
#   UNSET, the script auto-detects: every device under /sys/class/infiniband whose
#   ports/1/state is ACTIVE, minus any name listed in EXCLUDE_HCA (comma-separated
#   exact names -- same convention as nccl_bench/gpu_nic_topo.sh).
#
#   TRAP: an auto-detected set can include storage/management NICs. That is acceptable
#   for a *consistency* check (an outlier node is still caught), but the operator should
#   set COMPUTE_HCA_RE when they want the check scoped to the training fabric only.
head_ "B4. Compute HCA (IB/RoCE)"
cmd_ "ls /sys/class/infiniband   # all HCAs"
cmd_ "cat /sys/class/infiniband/<dev>/ports/1/{state,rate} ; cat .../fw_ver"
IB_DIR=/sys/class/infiniband
CDEVS=()
# Build the exclude map (exact-name, comma-separated -- same convention as gpu_nic_topo.sh)
declare -A _EXCLUDE_HCA=()
if [ -n "${EXCLUDE_HCA:-}" ]; then
    _IFS_BAK="${IFS}"; IFS=','
    for _h in ${EXCLUDE_HCA}; do
        _h="${_h// /}"; [ -n "$_h" ] && _EXCLUDE_HCA["$_h"]=1
    done
    IFS="${_IFS_BAK}"
fi
if [ -d "$IB_DIR" ]; then
    ALL_HCA=$(ls "$IB_DIR" 2>/dev/null | sort)
    HCA_LIST=$(printf '%s\n' "$ALL_HCA" | paste -sd, -)                 # all HCAs (reference)
    # Compute-HCA subset
    if [ -n "${COMPUTE_HCA_RE:-}" ]; then
        # Operator-scoped: anchored ERE match (original behaviour)
        mapfile -t CDEVS < <(printf '%s\n' "$ALL_HCA" | grep -E "^${COMPUTE_HCA_RE}$")
    else
        # Auto-detect: ACTIVE ports minus EXCLUDE_HCA
        while IFS= read -r d; do
            [ -z "$d" ] && continue
            st=$(cat "$IB_DIR/$d/ports/1/state" 2>/dev/null)
            echo "$st" | grep -q ACTIVE || continue
            [ -n "${_EXCLUDE_HCA[$d]:-}" ] && continue
            CDEVS+=("$d")
        done <<< "$ALL_HCA"
    fi
    CHCA_TOTAL=${#CDEVS[@]}
    CHCA_ACTIVE=0; crate=""; cfw=""; cll=""; cmtu=""; cpcie=""
    for d in "${CDEVS[@]}"; do
        st=$(cat "$IB_DIR/$d/ports/1/state" 2>/dev/null)
        echo "$st" | grep -q ACTIVE && CHCA_ACTIVE=$((CHCA_ACTIVE+1))
        crate="${crate}$(cat "$IB_DIR/$d/ports/1/rate" 2>/dev/null | awk '{print $1,$2}')\n"
        cfw="${cfw}$(cat "$IB_DIR/$d/fw_ver" 2>/dev/null)\n"
        # link_layer: Ethernet=RoCE / InfiniBand=IB
        cll="${cll}$(cat "$IB_DIR/$d/ports/1/link_layer" 2>/dev/null)\n"
        # bound netdev (for RoCE e.g. bond2) and its MTU
        nd=$(ls "$IB_DIR/$d/device/net" 2>/dev/null | head -1)
        cmtu="${cmtu}$(ip -o link show "$nd" 2>/dev/null | sed -nE 's/.* mtu ([0-9]+).*/\1/p')\n"
        # NIC PCIe link (Gen rate x width)
        bdf=$(basename "$(readlink -f "$IB_DIR/$d/device" 2>/dev/null)" 2>/dev/null)
        cpcie="${cpcie}$(lspci -s "$bdf" -vv 2>/dev/null | sed -nE 's/.*LnkSta:\s*Speed ([^,]+), Width (x[0-9]+).*/\1 \2/p' | head -1)\n"
    done
    CHCA_RATE=$(printf "%b" "$crate" | collapse)
    CHCA_FW=$(printf "%b" "$cfw" | collapse)
    CHCA_LIST=$(printf '%s\n' "${CDEVS[@]}" | paste -sd, -)
    _ll=$(printf "%b" "$cll" | collapse)
    case "$_ll" in Ethernet) FABRIC="RoCE";; InfiniBand) FABRIC="IB";; *) FABRIC="$_ll";; esac
    CHCA_MTU=$(printf "%b" "$cmtu" | collapse)
    CHCA_PCIE=$(printf "%b" "$cpcie" | collapse)
else
    HCA_LIST="none"; CHCA_LIST="none"; CHCA_TOTAL=0; CHCA_ACTIVE=0; CHCA_RATE="N/A"; CHCA_FW="N/A"
    FABRIC="N/A"; CHCA_MTU="N/A"; CHCA_PCIE="N/A"
fi
say "  fabric_type    : ${FABRIC}"
say "  compute_hca    : ${CHCA_LIST}"
say "  compute_active : ${CHCA_ACTIVE}/${CHCA_TOTAL}"
say "  compute_rate   : ${CHCA_RATE}"
say "  compute_mtu    : ${CHCA_MTU}"
say "  compute_pcie   : ${CHCA_PCIE}"
say "  compute_fw     : ${CHCA_FW}"
say "  all_hca(ref)   : ${HCA_LIST}"

# B5 OFED / fabricmanager / peermem
head_ "B5. OFED / FabricManager / peermem"
cmd_ "ofed_info -s"
cmd_ "cat /sys/module/nvidia_fabricmanager/version ; systemctl is-active nvidia-fabricmanager"
cmd_ "lsmod | grep -E 'nvidia_peermem|nv_peer_mem'"
OFED=$(ofed_info -s 2>/dev/null | sed 's/:$//' | head -1); [ -z "$OFED" ] && OFED="N/A"
FM_VER=$(cat /sys/module/nvidia_fabricmanager/version 2>/dev/null)
[ -z "$FM_VER" ] && FM_VER=$(nv-fabricmanager --version 2>/dev/null | awk '{print $NF}')
[ -z "$FM_VER" ] && FM_VER="N/A"
FM_ACTIVE=$(systemctl is-active nvidia-fabricmanager 2>/dev/null); [ -z "$FM_ACTIVE" ] && FM_ACTIVE="unknown"
if lsmod 2>/dev/null | grep -qE 'nvidia_peermem|nv_peer_mem'; then PEERMEM="loaded"; else PEERMEM="absent"; fi
say "  ofed           : ${OFED}"
say "  fabricmanager  : ${FM_VER} (${FM_ACTIVE})"
say "  peermem        : ${PEERMEM}"

# B6 kernel / OS / persistence / memlock / NUMA
head_ "B6. Kernel / OS / persistence / memlock"
cmd_ "uname -r ; grep PRETTY_NAME /etc/os-release ; ulimit -l ; lscpu | grep NUMA"
KERNEL=$(uname -r)
OS=$(sed -nE 's/^PRETTY_NAME="?([^"]*)"?/\1/p' /etc/os-release 2>/dev/null | head -1); [ -z "$OS" ] && OS="N/A"
if [ "$GPU_OK" -eq 1 ]; then
    PERSIST=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null | collapse)
else PERSIST="N/A"; fi
MEMLOCK=$(ulimit -l 2>/dev/null); [ -z "$MEMLOCK" ] && MEMLOCK="N/A"
NUMA_NODES=$(lscpu 2>/dev/null | awk -F: '/NUMA node\(s\)/{gsub(/ /,"",$2);print $2;exit}'); [ -z "$NUMA_NODES" ] && NUMA_NODES="N/A"
say "  kernel         : ${KERNEL}"
say "  os             : ${OS}"
say "  persistence    : ${PERSIST}"
say "  memlock(KB)     : ${MEMLOCK}"
say "  numa_nodes     : ${NUMA_NODES}"

# B7 bench container image
#   When BENCH_IMAGE_RE is set, grep that pattern to identify the bench image.
#   When unset, tag as not-scoped (do not match any product/registry name).
head_ "B7. Bench container image"
if [ -n "${BENCH_IMAGE_RE:-}" ]; then
    cmd_ "docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep '${BENCH_IMAGE_RE}'"
else
    cmd_ "# BENCH_IMAGE_RE unset -- bench image check is not scoped to any specific image"
fi
if has docker; then
    if [ -n "${BENCH_IMAGE_RE:-}" ]; then
        IMG=$(docker images --format '{{.Repository}}:{{.Tag}}|{{.ID}}' 2>/dev/null | grep -i "${BENCH_IMAGE_RE}" \
              | awk -F'|' '{print $2}' | sort -u | collapse)
    else
        IMG="not-scoped"
    fi
else IMG="N/A"; fi
[ -z "$IMG" ] && IMG="none"
say "  bench_image_id : ${IMG}"

# B8 Fabric / RoCE QoS / GPU-NIC affinity / NVLink (must-check before NCCL test)
head_ "B8. Fabric / RoCE / topo (NCCL)"
cmd_ "show_gids <dev>   # take the RoCE v2+IPv4 GID index (for NCCL_IB_GID_INDEX across subnets)"
cmd_ "mlnx_qos -i <netdev>   # trust(dscp/pcp) + PFC enabled priority (lossless)"
cmd_ "nvidia-smi topo -m   # GPU-NIC affinity (rail); nvidia-smi nvlink -s   # single-node NVLink"
FIRST_C="${CDEVS[0]:-}"
FIRST_ND=""
[ -n "$FIRST_C" ] && FIRST_ND=$(ls "$IB_DIR/$FIRST_C/device/net" 2>/dev/null | head -1)

# RoCE v2 + IPv4 GID index (NCCL_IB_GID_INDEX)
ROCE_GID_V2="N/A"
if [ "$FABRIC" = "RoCE" ] && [ -n "$FIRST_C" ]; then
    if has show_gids; then
        ROCE_GID_V2=$(show_gids "$FIRST_C" 2>/dev/null | awk '$0 ~ /v2/ && $5 ~ /[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/ {print $3; exit}')
    fi
    # Fallback: find "RoCE v2" with an IPv4-mapped gid (::ffff:) from sysfs gid_attrs/types
    if [ -z "$ROCE_GID_V2" ] || [ "$ROCE_GID_V2" = "N/A" ]; then
        for idx in 0 1 2 3 4 5; do
            t=$(cat "$IB_DIR/$FIRST_C/ports/1/gid_attrs/types/$idx" 2>/dev/null)
            g=$(cat "$IB_DIR/$FIRST_C/ports/1/gids/$idx" 2>/dev/null)
            if echo "$t" | grep -qi 'v2' && echo "$g" | grep -qi 'ffff:'; then ROCE_GID_V2="$idx"; break; fi
        done
    fi
fi
[ -z "$ROCE_GID_V2" ] && ROCE_GID_V2="N/A"

# RoCE lossless: trust mode + PFC enabled priority
ROCE_TRUST="N/A"; ROCE_PFC="N/A"
if [ "$FABRIC" = "RoCE" ] && has mlnx_qos && [ -n "$FIRST_ND" ]; then
    QOS=$(mlnx_qos -i "$FIRST_ND" 2>/dev/null)
    ROCE_TRUST=$(echo "$QOS" | sed -nE 's/.*trust state:\s*([a-zA-Z]+).*/\1/p' | head -1)
    # The PFC row is followed by a 0/1 sequence; grab the priority numbers where enabled=1
    ROCE_PFC=$(echo "$QOS" | awk '
        /priority +0 +1 +2/ {getline; if($1=="enabled"){o=""; for(i=2;i<=NF;i++) if($i==1) o=(o==""?(i-2):o","(i-2)); print (o==""?"none":o)}}' | head -1)
    [ -z "$ROCE_TRUST" ] && ROCE_TRUST="unknown"; [ -z "$ROCE_PFC" ] && ROCE_PFC="unknown"
fi

# GPU-NIC topology:
#   rail_pattern -- position-independent rail-structure signature (which GPUs share the
#                  same compute-card PXB rail); storage cards are never PXB and slot
#                  offsets do not affect it -> the real key for judging "is the wiring
#                  topology identical".
#   topo_sig     -- md5 of the full nvidia-smi topo -m matrix (includes storage cards,
#                  reference only; a diff here does NOT cause FAIL).
TOPO_SIG="N/A"; RAIL="N/A"; NVLINK="N/A"; NVSPEED="N/A"
if [ "$GPU_OK" -eq 1 ]; then
    TOPO=$(nvidia-smi topo -m 2>/dev/null | sed -E 's/\x1b\[[0-9;]*m//g')
    TOPO_SIG=$(printf '%s\n' "$TOPO" | awk '/^GPU[0-9]/{$1="";print}' | grep -oE 'NV[0-9]+|PIX|PXB|NODE|SYS|PHB' | md5sum 2>/dev/null | cut -c1-12)
    [ -z "$TOPO_SIG" ] && TOPO_SIG="N/A"
    RAIL=$(printf '%s\n' "$TOPO" | awk '
        $0 ~ /GPU0/ && $0 ~ /NIC0/ { for(i=1;i<=NF;i++) if($i ~ /^NIC[0-9]+$/) nicidx[++nn]=i; next }
        /^GPU[0-9]+/ {
            ng++; rank["PIX"]=1;rank["PXB"]=2;rank["PHB"]=3;rank["NODE"]=4;rank["SYS"]=5
            best=99; for(k=1;k<=nn;k++){ r=rank[$(nicidx[k]+1)]; if(r>0 && r<best) best=r }
            set=""; for(k=1;k<=nn;k++){ if(rank[$(nicidx[k]+1)]==best) set=set k"." }
            gset[ng]=set; if(best<minlvl||minlvl==0) minlvl=best
        }
        END{
            ln["1"]="PIX";ln["2"]="PXB";ln["3"]="PHB";ln["4"]="NODE";ln["5"]="SYS"
            grp=0; out=""
            for(i=1;i<=ng;i++){ if(!(gset[i] in id)) id[gset[i]]=grp++; out=(out==""?id[gset[i]]:out","id[gset[i]]) }
            print ln[minlvl""]" "out" ("grp"rail/"ng"gpu)"
        }')
    [ -z "$RAIL" ] && RAIL="N/A"
    if nvidia-smi nvlink -s >/dev/null 2>&1; then
        NVOUT=$(nvidia-smi nvlink -s 2>/dev/null)
        NVLINK=$(printf '%s\n' "$NVOUT" | grep -c 'Link .*: [0-9].*GB/s')
        [ "$NVLINK" = "0" ] && NVLINK="none/disabled"
        NVSPEED=$(printf '%s\n' "$NVOUT" | grep -oE '[0-9.]+ GB/s' | collapse)
        [ -z "$NVSPEED" ] && NVSPEED="N/A"
    fi
fi

# LLDP switch neighbour (the leaf switch + port each compute card uplinks to) -- best-effort.
#   lldptool/lldpctl is often not installed -> mark N/A (needs the network side to provide
#   a leaf-port-to-host map).
LLDP="N/A"
if [ "${#CDEVS[@]}" -gt 0 ]; then
    lld=""
    for d in "${CDEVS[@]}"; do
        nd=$(ls "$IB_DIR/$d/device/net" 2>/dev/null | head -1); [ -z "$nd" ] && continue
        if has lldptool; then
            nb=$(lldptool -t -i "$nd" -n 2>/dev/null | awk -F'[: ]+' '/System Name/{sn=$NF} /Port ID/{pid=$NF} END{if(sn||pid)print sn"/"pid}')
        elif has lldpctl; then
            nb=$(lldpctl "$nd" 2>/dev/null | awk '/SysName:/{sn=$2} /PortID:/{pid=$3} END{if(sn||pid)print sn"/"pid}')
        fi
        [ -n "${nb:-}" ] && lld="${lld}${nb}\n"
    done
    [ -n "$lld" ] && LLDP=$(printf "%b" "$lld" | collapse) || LLDP="N/A(network-side)"
fi

say "  fabric_type    : ${FABRIC}"
say "  roce_gid_v2idx : ${ROCE_GID_V2}"
say "  roce_trust     : ${ROCE_TRUST}"
say "  roce_pfc_prio  : ${ROCE_PFC}"
say "  gpu_nic_rail   : ${RAIL}   <- topology structure (position-independent, key item)"
say "  gpu_nic_topo   : ${TOPO_SIG}  (topo -m full signature, reference)"
say "  lldp_switch    : ${LLDP}"
say "  nvlink_links   : ${NVLINK}"
say "  nvlink_speed   : ${NVSPEED}"

# ---- output -----------------------------------------------------------------
emit_json() {
    printf '{'
    printf '"driver":"%s",'        "$DRIVER"
    printf '"cuda_api":"%s",'      "$CUDA_API"
    printf '"libcuda":"%s",'       "$LIBCUDA_VER"
    printf '"cuda_compat":"%s",'   "$CUDA_COMPAT"
    printf '"ecc_mode":"%s",'      "$ECC_MODE"
    printf '"ecc_uncorr":"%s",'    "$ECC_UNCORR"
    printf '"gpu_count":"%s",'     "$GPU_COUNT"
    printf '"sm_clock_max":"%s",'  "$SM_MAX"
    printf '"power_limit":"%s",'   "$PWR_LIMIT"
    printf '"hca_devices":"%s",'   "$HCA_LIST"
    printf '"fabric_type":"%s",'   "$FABRIC"
    printf '"compute_hca":"%s",'   "$CHCA_LIST"
    printf '"compute_active":"%s",' "${CHCA_ACTIVE}/${CHCA_TOTAL}"
    printf '"compute_rate":"%s",'  "$CHCA_RATE"
    printf '"compute_mtu":"%s",'   "$CHCA_MTU"
    printf '"compute_pcie":"%s",'  "$CHCA_PCIE"
    printf '"roce_gid_v2idx":"%s",' "$ROCE_GID_V2"
    printf '"roce_trust":"%s",'    "$ROCE_TRUST"
    printf '"roce_pfc_prio":"%s",' "$ROCE_PFC"
    printf '"gpu_nic_rail":"%s",'  "$RAIL"
    printf '"gpu_nic_topo":"%s",'  "$TOPO_SIG"
    printf '"lldp_switch":"%s",'   "$LLDP"
    printf '"nvlink_links":"%s",'  "$NVLINK"
    printf '"nvlink_speed":"%s",'  "$NVSPEED"
    printf '"compute_fw":"%s",'    "$CHCA_FW"
    printf '"ofed":"%s",'          "$OFED"
    printf '"fabricmanager":"%s",' "$FM_VER"
    printf '"fm_active":"%s",'     "$FM_ACTIVE"
    printf '"peermem":"%s",'       "$PEERMEM"
    printf '"kernel":"%s",'        "$KERNEL"
    printf '"os":"%s",'            "$OS"
    printf '"persistence":"%s",'   "$PERSIST"
    printf '"memlock":"%s",'       "$MEMLOCK"
    printf '"numa_nodes":"%s",'    "$NUMA_NODES"
    printf '"bench_image_id":"%s"' "$IMG"
    printf '}\n'
}

if [ "$JSON_MODE" -eq 1 ]; then
    emit_json
else
    head_ "Runtime-state summary"
    say "  B1-B7 collected above. JSON: bash $0 -j"
fi

if [ -n "$OUT_FILE" ]; then
    JSON_MODE=1 emit_json > "$OUT_FILE" 2>/dev/null || emit_json > "$OUT_FILE"
    say "  written to: $OUT_FILE"
fi
