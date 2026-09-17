#!/bin/bash
# ============================================================================
# cluster_spec_check.sh
#   Cluster-level server-spec consistency check.
#   Flow:
#     1) Read hosts.txt (one IP/host per line)
#     2) Pipe collect_server_spec.sh -j to each node over plain ssh (stdin) and run it
#     3) Collect every node's single-line JSON result
#     4) For each spec field take the most frequent value as the "baseline" and flag
#        the nodes that differ
#     5) Output a consistency matrix + diff report (for acceptance / cluster delivery audit)
#
# Transport: plain ssh only. The collector script is fed to each node's bash over
# stdin, so no shared filesystem and no parallel-ssh fan-out tool is required.
#
# Usage:
#     bash cluster_spec_check.sh -H hosts.txt
#     bash cluster_spec_check.sh -H hosts.txt -u root
#     bash cluster_spec_check.sh -H hosts.txt -p 200        # -p = parallelism (concurrency)
#     bash cluster_spec_check.sh -H hosts.txt -P 2222       # -P = SSH port
#     bash cluster_spec_check.sh -H hosts.txt -o cluster_spec.csv  # also output CSV
#     bash cluster_spec_check.sh -H hosts.txt -k            # keep per-node raw JSON results
#
# Dependencies: ssh, awk, sed (passwordless SSH must be configured).
# ============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_server_spec.sh"

HOSTS_FILE=""
SSH_USER="root"
SSH_PORT="22"
PARALLEL=64
CSV_OUT=""
KEEP_RAW=0    # -k: keep per-node raw JSON / text results

while getopts "H:u:p:P:o:kh" opt; do
    case $opt in
        H) HOSTS_FILE="$OPTARG" ;;
        u) SSH_USER="$OPTARG" ;;
        p) PARALLEL="$OPTARG" ;;
        P) SSH_PORT="$OPTARG" ;;
        o) CSV_OUT="$OPTARG" ;;
        k) KEEP_RAW=1 ;;
        h)
            sed -n '2,25p' "$0"
            exit 0
            ;;
    esac
done

# colors
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
log_ok()    { echo -e "${C_GRN}[ OK ]${C_RST}  $*"; }
log_diff()  { echo -e "${C_RED}[DIFF]${C_RST}  $*"; }

# ---------------------------- arg check ----------------------------
if [ -z "$HOSTS_FILE" ] || [ ! -f "$HOSTS_FILE" ]; then
    log_error "specify a valid hosts file with -H (one IP/host per line)"
    exit 1
fi
if [ ! -f "$COLLECT_SCRIPT" ]; then
    log_error "single-node collector script not found: $COLLECT_SCRIPT"
    exit 1
fi
REQ_TOOLS="ssh awk sed"
for cmd in $REQ_TOOLS; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log_error "missing required tool: $cmd"
        exit 1
    fi
done

# Read node list (drop blank lines / comments)
mapfile -t NODES < <(grep -vE '^\s*(#|$)' "$HOSTS_FILE" | awk '{print $1}')
NODE_CNT=${#NODES[@]}
if [ "$NODE_CNT" -eq 0 ]; then
    log_error "no valid nodes in hosts file"
    exit 1
fi

log_head "Cluster spec consistency check"
log_info "nodes            : ${NODE_CNT}"
log_info "SSH user          : ${SSH_USER}"
log_info "SSH port          : ${SSH_PORT}"
log_info "parallelism       : ${PARALLEL}"
log_info "collector         : ${COLLECT_SCRIPT}"

# ---------------------------- Step 1. collect ----------------------------
log_head "Step 1. Collect JSON spec from each node"

WORK_DIR=$(mktemp -d -t cluster_spec_XXXXXX)
trap '[ "$KEEP_RAW" -eq 0 ] && rm -rf "$WORK_DIR"' EXIT
log_info "work dir          : ${WORK_DIR}"

OUT_DIR="${WORK_DIR}/out"
mkdir -p "$OUT_DIR"

log_info "transport         : plain ssh (script piped over stdin; no shared fs)"

# When the SSH user is not root, prefix the remote command with sudo so dmidecode /
# lsblk / ethtool fields are complete (same behaviour the original parallel path used).
SUDO_PREFIX=""
if [ "$SSH_USER" != "root" ]; then SUDO_PREFIX="sudo "; fi

log_head "Step 2. Concurrent JSON collection per node"
pids=()
for h in "${NODES[@]}"; do
    (
        timeout 180 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
            -p "${SSH_PORT}" "${SSH_USER}@${h}" "${SUDO_PREFIX}bash -s -- -j" \
            < "$COLLECT_SCRIPT" > "$OUT_DIR/$h" 2>/dev/null
    ) &
    pids+=($!)
    # simple concurrency cap: never more than $PARALLEL in flight
    if [ "${#pids[@]}" -ge "$PARALLEL" ]; then
        for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
        pids=()
    fi
done
for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done

# ---------------------------- Step 3. parse + consistency compare ----------------------------
log_head "Step 3. Parse each node JSON"

# Field list (strictly aligned with collect_server_spec.sh -j keys)
FIELDS=(gpu cpu memory storage_os storage_data nic_compute nic_storage)
FIELD_LABELS=(GPU CPU Memory "Storage(OS)" "Storage(Data)" "NIC compute" "NIC storage")

# host -> field -> value
declare -A VAL    # key="$host|$field"
declare -A REACHABLE   # host -> 1/0

for h in "${NODES[@]}"; do
    f="${OUT_DIR}/${h}"
    if [ ! -s "$f" ]; then
        REACHABLE["$h"]=0
        log_warn "node ${h}: no output (unreachable / script failed)"
        continue
    fi
    # Take the last non-empty line (should be JSON)
    line=$(grep -E '^\{' "$f" | tail -n1)
    if [ -z "$line" ]; then
        REACHABLE["$h"]=0
        log_warn "node ${h}: no JSON found in output"
        continue
    fi
    REACHABLE["$h"]=1
    # Minimal JSON extraction: sed grabs "key":"value" (values contain no unescaped ")
    for fld in "${FIELDS[@]}"; do
        v=$(echo "$line" | sed -nE "s/.*\"${fld}\":\"([^\"]*)\".*/\1/p")
        VAL["${h}|${fld}"]="$v"
    done
done

# ---------------------------- Step 4. consistency matrix ----------------------------
log_head "Step 4. Consistency results"

OVERALL_OK=1
declare -A BASELINE   # field -> baseline value

# Compute the mode of each field as the baseline
for i in "${!FIELDS[@]}"; do
    fld="${FIELDS[$i]}"
    label="${FIELD_LABELS[$i]}"
    # Collect this field's values across reachable nodes only
    mapfile -t _vals < <(
        for h in "${NODES[@]}"; do
            [ "${REACHABLE[$h]:-0}" -eq 1 ] || continue
            printf '%s\n' "${VAL["${h}|${fld}"]}"
        done
    )
    if [ "${#_vals[@]}" -eq 0 ]; then
        BASELINE["$fld"]="<no data>"
        continue
    fi
    base=$(printf '%s\n' "${_vals[@]}" | sort | uniq -c | sort -rn | head -n1 | sed -E 's/^[[:space:]]*[0-9]+[[:space:]]+//')
    BASELINE["$fld"]="$base"

    # List nodes that differ from baseline
    diff_hosts=()
    for h in "${NODES[@]}"; do
        [ "${REACHABLE[$h]:-0}" -eq 1 ] || continue
        if [ "${VAL["${h}|${fld}"]}" != "$base" ]; then
            diff_hosts+=("$h")
        fi
    done

    if [ "${#diff_hosts[@]}" -eq 0 ]; then
        log_ok "${label}  all agree   ->  ${base}"
    else
        OVERALL_OK=0
        log_diff "${label}  diff present"
        echo -e "       baseline (${C_GRN}$(( NODE_CNT - ${#diff_hosts[@]} - $(echo "${!REACHABLE[@]}" | tr ' ' '\n' | xargs -I{} sh -c '[ "${REACHABLE[{}]}" = "0" ] && echo 1' 2>/dev/null | wc -l) ))${C_RST} nodes): ${base}"
        for h in "${diff_hosts[@]}"; do
            echo -e "       ${C_RED}x ${h}${C_RST} : ${VAL["${h}|${fld}"]}"
        done
    fi
done

# Unreachable nodes
UNREACHABLE_HOSTS=()
for h in "${NODES[@]}"; do
    [ "${REACHABLE[$h]:-0}" -eq 0 ] && UNREACHABLE_HOSTS+=("$h")
done
if [ "${#UNREACHABLE_HOSTS[@]}" -gt 0 ]; then
    OVERALL_OK=0
    log_diff "unreachable nodes (${#UNREACHABLE_HOSTS[@]}/${NODE_CNT}):"
    for h in "${UNREACHABLE_HOSTS[@]}"; do
        echo -e "       ${C_RED}x ${h}${C_RST}"
    done
fi

# ---------------------------- Step 5. CSV export (optional) ----------------------------
if [ -n "$CSV_OUT" ]; then
    log_head "Step 5. Export CSV: ${CSV_OUT}"
    {
        # header
        printf 'host,reachable'
        for label in "${FIELD_LABELS[@]}"; do printf ',%s' "$label"; done
        printf '\n'
        # one row per node
        for h in "${NODES[@]}"; do
            r="${REACHABLE[$h]:-0}"
            printf '%s,%s' "$h" "$r"
            for fld in "${FIELDS[@]}"; do
                v="${VAL["${h}|${fld}"]:-}"
                # CSV escape: contains , or " -> quote + " -> ""
                if [[ "$v" == *,* || "$v" == *\"* ]]; then
                    v=${v//\"/\"\"}
                    printf ',"%s"' "$v"
                else
                    printf ',%s' "$v"
                fi
            done
            printf '\n'
        done
    } > "$CSV_OUT"
    log_info "CSV written to: ${CSV_OUT}"
fi

# Keep raw results
if [ "$KEEP_RAW" -eq 1 ]; then
    log_info "raw JSON output kept in: ${OUT_DIR}"
fi

# ---------------------------- Step 6. summary ----------------------------
log_head "Step 6. Summary"
if [ "$OVERALL_OK" -eq 1 ]; then
    echo -e "${C_GRN}${C_BLD}OK: all cluster nodes have identical specs (${NODE_CNT} nodes)${C_RST}"
    exit 0
else
    echo -e "${C_RED}${C_BLD}FAIL: cluster has inconsistent or unreachable nodes; investigate the diff list above${C_RST}"
    exit 2
fi
