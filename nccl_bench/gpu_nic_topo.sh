#!/bin/bash
# ============================================================================
# gpu_nic_topo.sh
#   Parse `nvidia-smi topo -m` plus `ibdev2netdev` and report:
#     - the best-affinity RDMA HCA per GPU (PIX > PXB > PHB > NODE > SYS)
#         * when several candidates share the best level, pick the primary by
#           (rate desc, HCA tail number desc)
#         * EXCLUDE_HCA=mlx5_1,mlx5_6 keeps management/storage cards out
#     - ALL same-level candidate HCAs per GPU, with their rates (real topology)
#     - the Linux netdev name and link state behind each HCA
#
# Two ways to use it:
#   1) Run it:    bash gpu_nic_topo.sh [--json|--csv|--nccl-hca|--verbose]
#   2) Source it: source gpu_nic_topo.sh  then use:
#          GPU_HCA[i]            = mlx5_x     (primary affine HCA of GPU i)
#          GPU_NETDEV[i]         = ensXXnpY   (its Linux netdev)
#          GPU_HCA_LEVEL[i]      = PIX/PXB/NODE  (affinity level)
#          GPU_HCA_ALL[i]        = mlx5_1,mlx5_101  (all same-level candidates)
#          GPU_HCA_ALL_DETAIL[i] = mlx5_1(25G),mlx5_101(400G)  (with rates)
#          NCCL_IB_HCA_STR       = "mlx5_0,mlx5_1,..."  (GPU order, deduped,
#                                                          Down ports dropped)
# ============================================================================

# ------------------- dependency check -------------------
_need() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "[ERROR] missing command: $1" >&2
        return 1
    }
}
_need nvidia-smi || exit 1
_need ibdev2netdev || {
    echo "[NOTE] ibdev2netdev not found — GPU<->HCA affinity only, no link state" >&2
}

# ------------------- user configurable: HCA exclusion list -------------------
# EXCLUDE_HCA=mlx5_1,mlx5_6 excludes management/storage NICs that are not
# carrying training traffic.
declare -A _EXCLUDE_HCA=()
if [ -n "${EXCLUDE_HCA:-}" ]; then
    _IFS_BAK="${IFS}"
    IFS=','
    for _h in ${EXCLUDE_HCA}; do
        _h="${_h// /}"
        [ -n "${_h}" ] && _EXCLUDE_HCA["${_h}"]=1
    done
    IFS="${_IFS_BAK}"
fi

# ------------------- HCA rate in Gb/s (0 if unreadable) -------------------
_hca_rate_gbps() {
    local hca="$1"
    local rate_str=""
    if [ -r "/sys/class/infiniband/${hca}/ports/1/rate" ]; then
        rate_str="$(cat /sys/class/infiniband/${hca}/ports/1/rate 2>/dev/null)"
    fi
    if [ -z "${rate_str}" ] && command -v ibv_devinfo >/dev/null 2>&1; then
        rate_str="$(ibv_devinfo -d "${hca}" 2>/dev/null | awk '/[[:space:]]rate:/{print $2; exit}')"
    fi
    echo "${rate_str}" | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/){print $i; exit}}' | head -1
}

# Cache HCA rates (each lookup shells out)
declare -A _HCA_RATE_CACHE=()
_get_hca_rate() {
    local hca="$1"
    [ -z "${hca}" ] && { echo "0"; return; }
    if [ -z "${_HCA_RATE_CACHE[${hca}]+set}" ]; then
        local r
        r="$(_hca_rate_gbps "${hca}")"
        _HCA_RATE_CACHE["${hca}"]="${r:-0}"
    fi
    echo "${_HCA_RATE_CACHE[${hca}]}"
}

# ------------------- affinity ranking (lower is better) -------------------
_level_rank() {
    case "$1" in
        PIX)  echo 0 ;;
        PXB)  echo 1 ;;
        PHB)  echo 2 ;;
        NODE) echo 3 ;;
        SYS)  echo 4 ;;
        *)    echo 9 ;;
    esac
}

# ------------------- 1. parse nvidia-smi topo -m -------------------
TOPO_RAW="$(nvidia-smi topo -m 2>/dev/null)"
if [ -z "${TOPO_RAW}" ]; then
    echo "[ERROR] nvidia-smi topo -m produced no output" >&2
    exit 1
fi

# Find the header line and the NICx column indices
HEADER_LINE="$(echo "${TOPO_RAW}" | awk '/^[[:space:]]+GPU0[[:space:]]/{print;exit}')"
if [ -z "${HEADER_LINE}" ]; then
    HEADER_LINE="$(echo "${TOPO_RAW}" | awk '/GPU0/ && /NIC0/ {print;exit}')"
fi
if [ -z "${HEADER_LINE}" ]; then
    echo "[ERROR] cannot parse the nvidia-smi topo -m header" >&2
    exit 1
fi

read -ra HEADER_TOKENS <<< "${HEADER_LINE}"
declare -a NIC_NAMES=()
declare -a NIC_TOKEN_IDX=()
_nic_re='^NIC[0-9]+$'
for i in "${!HEADER_TOKENS[@]}"; do
    tok="${HEADER_TOKENS[$i]}"
    if [[ "${tok}" =~ ${_nic_re} ]]; then
        NIC_NAMES+=("${tok}")
        NIC_TOKEN_IDX+=($((i + 1)))
    fi
done

if [ "${#NIC_NAMES[@]}" -eq 0 ]; then
    echo "[WARN] no NIC columns in nvidia-smi topo -m (no RDMA NIC, or an old driver)" >&2
    echo "[DEBUG] header line: ${HEADER_LINE}" >&2
fi

# Parse the NIC legend: "NIC0: mlx5_0"
declare -A NIC_TO_HCA=()
_legend_re='^[[:space:]]*(NIC[0-9]+):[[:space:]]*([[:alnum:]_]+)'
while IFS= read -r line; do
    if [[ "${line}" =~ ${_legend_re} ]]; then
        NIC_TO_HCA["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
    fi
done <<< "${TOPO_RAW}"

# Parse the GPUx rows
declare -a GPU_LIST=()
declare -a GPU_HCA=()             # primary HCA (after ranking)
declare -a GPU_NETDEV=()
declare -a GPU_HCA_LEVEL=()
declare -a GPU_HCA_ALL=()         # all same-level candidate HCAs, comma separated
declare -a GPU_HCA_ALL_DETAIL=()  # same, with rates

_gpu_row_re='^[[:space:]]*GPU[0-9]+[[:space:]]'
while IFS= read -r line; do
    [[ "${line}" =~ ${_gpu_row_re} ]] || continue

    read -ra COLS <<< "${line}"
    gpu_name="${COLS[0]}"
    GPU_LIST+=("${gpu_name}")

    best_level="ZZ"
    best_rank=99
    best_same_level=""   # all NICs at the best level

    # Pass 1: find the best affinity level and collect every NIC at it
    for k in "${!NIC_NAMES[@]}"; do
        nic="${NIC_NAMES[$k]}"
        idx="${NIC_TOKEN_IDX[$k]}"
        lvl="${COLS[$idx]}"
        [ -z "${lvl}" ] && continue
        rank=$(_level_rank "${lvl}")
        if [ "${rank}" -lt "${best_rank}" ]; then
            best_rank="${rank}"
            best_level="${lvl}"
            best_same_level="${nic}"
        elif [ "${rank}" -eq "${best_rank}" ]; then
            best_same_level="${best_same_level},${nic}"
        fi
    done

    # Pass 2: pick the primary by (rate desc, HCA tail desc), and record every
    # candidate with its rate
    best_hca=""
    best_rate=-1
    best_tail=-1
    same_hca=""
    same_hca_detail=""

    if [ -n "${best_same_level}" ]; then
        IFS=',' read -ra _cands <<< "${best_same_level}"
        for c in "${_cands[@]}"; do
            [ -z "${c}" ] && continue
            h="${NIC_TO_HCA[${c}]:-}"
            [ -z "${h}" ] && continue

            rate="$(_get_hca_rate "${h}")"
            rate="${rate:-0}"

            # Record all candidates (with rate) so the real topology is visible
            hca_with_rate="${h}(${rate}G)"
            if [ -z "${same_hca}" ]; then
                same_hca="${h}"
                same_hca_detail="${hca_with_rate}"
            else
                same_hca="${same_hca},${h}"
                same_hca_detail="${same_hca_detail},${hca_with_rate}"
            fi

            # Excluded HCAs may still be listed, but never selected as primary
            [ -n "${_EXCLUDE_HCA[${h}]:-}" ] && continue

            tail_num="$(echo "${h}" | grep -oE '[0-9]+$')"
            tail_num="${tail_num:-0}"

            # Ranking: higher rate wins; ties go to the larger tail number
            if [ "${rate}" -gt "${best_rate}" ] \
               || { [ "${rate}" -eq "${best_rate}" ] && [ "${tail_num}" -gt "${best_tail}" ]; }; then
                best_rate="${rate}"
                best_tail="${tail_num}"
                best_hca="${h}"
            fi
        done
    fi

    # Fallback: if every candidate was excluded, take the first one anyway
    if [ -z "${best_hca}" ] && [ -n "${same_hca}" ]; then
        best_hca="${same_hca%%,*}"
    fi

    GPU_HCA+=("${best_hca}")
    GPU_HCA_LEVEL+=("${best_level}")
    GPU_HCA_ALL+=("${same_hca}")
    GPU_HCA_ALL_DETAIL+=("${same_hca_detail}")
done <<< "${TOPO_RAW}"

# ------------------- 2. parse ibdev2netdev -------------------
declare -A HCA_TO_NETDEV=()
declare -A HCA_TO_STATE=()
_ib_re='^([[:alnum:]_]+)[[:space:]]+port[[:space:]]+[0-9]+[[:space:]]+==>[[:space:]]+([^[:space:]]+)[[:space:]]+\(([^)]+)\)'
if command -v ibdev2netdev >/dev/null 2>&1; then
    while IFS= read -r line; do
        if [[ "${line}" =~ ${_ib_re} ]]; then
            HCA_TO_NETDEV["${BASH_REMATCH[1]}"]="${BASH_REMATCH[2]}"
            HCA_TO_STATE["${BASH_REMATCH[1]}"]="${BASH_REMATCH[3]}"
        fi
    done < <(ibdev2netdev 2>/dev/null)
fi

# Fill GPU_NETDEV
for i in "${!GPU_LIST[@]}"; do
    hca="${GPU_HCA[$i]}"
    GPU_NETDEV+=("${HCA_TO_NETDEV[${hca}]:-}")
done

# ------------------- 3. build the NCCL_IB_HCA string -------------------
# IMPORTANT: use every candidate at the GPU's best affinity level
# (GPU_HCA_ALL), NOT just the single selected primary HCA.
# Under a PXB topology a pair of GPUs shares two HCAs; picking one primary per
# GPU and then de-duplicating would keep only half the rails (8 -> 4) and halve
# cross-node all2all/allgather bandwidth. Collecting all same-level candidates
# fills the rail set correctly under both PXB and PIX.
NCCL_IB_HCA_STR=""
declare -A _seen=()
for i in "${!GPU_LIST[@]}"; do
    IFS=',' read -ra _hca_cands <<< "${GPU_HCA_ALL[$i]}"
    for hca in "${_hca_cands[@]}"; do
        [ -z "${hca}" ] && continue
        [ -n "${_seen[${hca}]:-}" ] && continue
        [ -n "${_EXCLUDE_HCA[${hca}]:-}" ] && continue
        state="${HCA_TO_STATE[${hca}]:-Unknown}"
        [ "${state}" = "Down" ] && continue
        _seen["${hca}"]=1
        if [ -z "${NCCL_IB_HCA_STR}" ]; then
            NCCL_IB_HCA_STR="${hca}"
        else
            NCCL_IB_HCA_STR="${NCCL_IB_HCA_STR},${hca}"
        fi
    done
done

# ------------------- 4. output -------------------
_is_sourced() {
    [ "${BASH_SOURCE[0]}" != "${0}" ]
}

_print_table() {
    printf "%-6s %-10s %-9s %-6s %-14s %-6s  %s\n" \
        "GPU" "MAIN_HCA" "AFFINITY" "RATE" "NETDEV" "LINK" "ALL_PIX_NICS(rate)"
    printf "%-6s %-10s %-9s %-6s %-14s %-6s  %s\n" \
        "----" "--------" "--------" "----" "------" "----" "------------------"
    for i in "${!GPU_LIST[@]}"; do
        main_hca="${GPU_HCA[$i]:--}"
        if [ "${main_hca}" != "-" ]; then
            main_rate="$(_get_hca_rate "${main_hca}")G"
        else
            main_rate="-"
        fi
        printf "%-6s %-10s %-9s %-6s %-14s %-6s  %s\n" \
            "${GPU_LIST[$i]}" \
            "${main_hca}" \
            "${GPU_HCA_LEVEL[$i]:--}" \
            "${main_rate}" \
            "${GPU_NETDEV[$i]:--}" \
            "${HCA_TO_STATE[${GPU_HCA[$i]}]:--}" \
            "${GPU_HCA_ALL_DETAIL[$i]:--}"
    done
}

_print_json() {
    printf '{\n  "gpu_nic_mapping": [\n'
    for i in "${!GPU_LIST[@]}"; do
        sep=","
        [ "$i" -eq $((${#GPU_LIST[@]} - 1)) ] && sep=""
        main_hca="${GPU_HCA[$i]}"
        main_rate="$(_get_hca_rate "${main_hca}")"
        printf '    {"gpu":"%s","main_hca":"%s","rate_gbps":%s,"affinity":"%s","netdev":"%s","link":"%s","all_pix_nics":"%s"}%s\n' \
            "${GPU_LIST[$i]}" \
            "${main_hca}" \
            "${main_rate:-0}" \
            "${GPU_HCA_LEVEL[$i]}" \
            "${GPU_NETDEV[$i]}" \
            "${HCA_TO_STATE[${GPU_HCA[$i]}]:-}" \
            "${GPU_HCA_ALL_DETAIL[$i]}" \
            "${sep}"
    done
    printf '  ],\n  "nccl_ib_hca":"%s"\n}\n' "${NCCL_IB_HCA_STR}"
}

_print_csv() {
    echo "gpu,main_hca,rate_gbps,affinity,netdev,link,all_pix_nics"
    for i in "${!GPU_LIST[@]}"; do
        main_hca="${GPU_HCA[$i]}"
        main_rate="$(_get_hca_rate "${main_hca}")"
        echo "${GPU_LIST[$i]},${main_hca},${main_rate:-0},${GPU_HCA_LEVEL[$i]},${GPU_NETDEV[$i]},${HCA_TO_STATE[${GPU_HCA[$i]}]:-},\"${GPU_HCA_ALL_DETAIL[$i]}\""
    done
}

if _is_sourced; then
    export NCCL_IB_HCA_STR
    return 0 2>/dev/null || true
fi

MODE="table"
case "${1:-}" in
    --json)     MODE="json" ;;
    --csv)      MODE="csv"  ;;
    --nccl-hca) MODE="hca"  ;;
    --verbose|-v) MODE="verbose" ;;
    "" ) ;;
    -h|--help)
        cat <<EOF
Usage: bash $0 [option]
  (no option)   table of GPU <-> HCA <-> netdev affinity (default)
  --json        JSON output
  --csv         CSV output
  --nccl-hca    only the comma-separated string for NCCL_IB_HCA
  --verbose     table plus the NCCL_IB_HCA line
  -h, --help    this help

Environment:
  EXCLUDE_HCA   HCAs to exclude (management/storage NICs), comma separated
                example: EXCLUDE_HCA=mlx5_1,mlx5_6 bash $0

Primary-HCA ranking within the best affinity level:
  1. drop anything in EXCLUDE_HCA
  2. higher link rate first (400G > 100G > 25G)
  3. on equal rate, larger HCA tail number first (mlx5_101 > mlx5_1)

Columns:
  MAIN_HCA           the selected primary affine HCA (for NCCL_IB_HCA)
  RATE               its link rate in Gb/s
  ALL_PIX_NICS(rate) every candidate at that affinity level, with rates

When sourced:
  source $0
  echo "\${NCCL_IB_HCA_STR}"
EOF
        exit 0
        ;;
    *)
        echo "[ERROR] unknown argument: $1 (use -h for help)" >&2
        exit 1
        ;;
esac

case "${MODE}" in
    table)   _print_table ;;
    json)    _print_json  ;;
    csv)     _print_csv   ;;
    hca)     echo "${NCCL_IB_HCA_STR}" ;;
    verbose)
        _print_table
        echo ""
        echo "NCCL_IB_HCA=${NCCL_IB_HCA_STR}"
        ;;
esac
