#!/bin/bash
# ============================================================================
# cluster_cublas_percard.sh — Fleet-wide per-card cuBLAS GEMM outlier detection
# ----------------------------------------------------------------------------
# Runs cublasMatmulBench on every GPU of every node in the host list,
# builds a (node x card x precision) matrix, and performs fleet-wide
# outlier detection + baseline comparison.
#
# Transport: plain ssh, parallelised via background jobs. No conda, no pssh
# required. A shared filesystem is NOT required for the runner script:
# _percard_remote.sh is piped over stdin (ssh host 'bash -s' < _percard_remote.sh).
# The cublas binaries must be accessible on each node at SHARE_DIR (set -s).
#
# Usage:
#   bash cluster_cublas_percard.sh -H hosts.txt [options]
#   bash cluster_cublas_percard.sh -H hosts.txt --preflight
#   bash cluster_cublas_percard.sh -H hosts.txt -y
#
# Required:
#   -H <hosts_file>   Host list file (one host per line, '#' comments allowed)
#   -i <image>        Docker image (provides CUDA runtime; binaries ship in repo)
#
# Options:
#   -P <port>          SSH port (default: 22)
#   -u <user>           SSH user (default: root)
#   -g <gpu_type>       GPU type: H800/H20/H200/H100/B200/B300/GB200/GB300/A800/L20
#   -s <share_dir>       Path to the cublas_bench dir on remote nodes
#                        (default: current directory; must be accessible on each node)
#   -n <gpus_per_node>   GPUs per node (default: 8)
#   -o <output_dir>      Where to write per-card logs on this machine
#                        (default: <repo>/results/cublas)
#   --preflight          Only do connectivity + image precheck, then exit
#   -y, --yes            Non-interactive (skip confirmation prompt)
#   -h, --help           Show this help
#
# Environment variables (override flags):
#   HOSTS, PORT, SSH_USER, GPU_TYPE, IMAGE, SHARE_DIR, GPUS_PER_NODE, OUT_DIR
# ============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Remember where the caller was: this script cds into its own directory, which
# would otherwise break caller-relative arguments like `-H ../preflight/hosts.txt`.
ORIG_PWD="$PWD"
cd "$SCRIPT_DIR"

# ---------- defaults ----------
PORT="${PORT:-22}"
SSH_USER="${SSH_USER:-root}"
GPU_TYPE="${GPU_TYPE:-H800}"
SHARE_DIR="${SHARE_DIR:-$SCRIPT_DIR}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
IMAGE="${IMAGE:-}"
HOSTS="${HOSTS:-}"
# Where the per-card logs land on THIS machine. Defaults into the repository's
# results/ tree so tools/collect_results.sh and tools/summarize.py find it
# without any manual moving of files.
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/../results/cublas}"
FORCE=0
PREFLIGHT_ONLY=0

# ---------- argument parsing ----------
usage() {
    # Print the leading comment block (skipping the shebang) as usage.
    # Use an absolute path: this script cds to its own directory above, so the
    # caller-relative "$0" may no longer resolve.
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' \
        "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -H) HOSTS="$2"; shift 2 ;;
        -P) PORT="$2"; shift 2 ;;
        -u) SSH_USER="$2"; shift 2 ;;
        -g) GPU_TYPE="$2"; shift 2 ;;
        -i) IMAGE="$2"; shift 2 ;;
        -s) SHARE_DIR="$2"; shift 2 ;;
        -o) OUT_DIR="$2"; shift 2 ;;
        -n) GPUS_PER_NODE="$2"; shift 2 ;;
        -y|--yes) FORCE=1; shift ;;
        --preflight|--preflight-only) PREFLIGHT_ONLY=1; shift ;;
        -h|--help) usage ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------- resolve caller-relative paths ----------
# Resolve BEFORE the cd above could matter: -H and -o are local paths given
# relative to wherever the caller stands. SHARE_DIR is deliberately NOT resolved
# — it is a path on the REMOTE nodes, not here.
case "$HOSTS" in
    /*) ;;
    *) HOSTS="$ORIG_PWD/$HOSTS" ;;
esac
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ORIG_PWD/$OUT_DIR" ;;
esac

# ---------- validate required args ----------
if [[ -z "$HOSTS" ]]; then
    echo "[ERROR] -H <hosts_file> is required." >&2
    echo "Usage: bash $(basename "$0") -H <hosts_file> -i <image> [options]" >&2
    exit 2
fi
if [[ ! -s "$HOSTS" ]]; then
    echo "[ERROR] hosts file is missing or empty: $HOSTS" >&2
    exit 2
fi
if [[ -z "$IMAGE" ]]; then
    echo "[ERROR] -i <image> (or IMAGE env var) is required." >&2
    echo "The image provides the CUDA runtime; the cublasMatmulBench binaries ship in this repo." >&2
    exit 2
fi

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes"

log(){ echo -e "\033[1;34m[$(date +%H:%M:%S)]\033[0m $*"; }
die(){ echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }
phase(){ echo -e "\n\033[1;36m========== $* ==========\033[0m"; }

# ---------- parallel ssh: run a command string on all hosts ----------
# Output format mimics pssh -i: [SUCCESS]/[FAILURE] + raw output per host.
_ssh_parallel_cmd() {
    local cmd="$1" tmpd pids=() h out rc
    tmpd=$(mktemp -d)
    while read -r h _; do
        case "$h" in ''|\#*) continue;; esac
        (
            out=$(ssh $SSH_OPTS -p "$PORT" -l "$SSH_USER" "$h" "$cmd" 2>&1); rc=$?
            { if [ "$rc" -eq 0 ]; then echo "[SUCCESS] $h"; else echo "[FAILURE] $h"; fi; echo "$out"; } > "$tmpd/$h"
        ) &
        pids+=($!)
    done < <(grep -vE '^[[:space:]]*(#|$)' "$HOSTS" | awk '{print $1}')
    for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
    cat "$tmpd"/* 2>/dev/null
    rm -rf "$tmpd"
}

# ---------- parallel ssh: pipe a script via stdin to all hosts ----------
_ssh_parallel_pipe() {
    local script="$1" env_prefix="$2" tmpd pids=() h out rc
    tmpd=$(mktemp -d)
    while read -r h _; do
        case "$h" in ''|\#*) continue;; esac
        (
            out=$(ssh $SSH_OPTS -p "$PORT" -l "$SSH_USER" "$h" "$env_prefix NODE_ID='$h' bash -s" < "$script" 2>&1); rc=$?
            { if [ "$rc" -eq 0 ]; then echo "[SUCCESS] $h"; else echo "[FAILURE] $h"; fi; echo "$out"; } > "$tmpd/$h"
        ) &
        pids+=($!)
    done < <(grep -vE '^[[:space:]]*(#|$)' "$HOSTS" | awk '{print $1}')
    for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
    cat "$tmpd"/* 2>/dev/null
    rm -rf "$tmpd"
}

NHOSTS=$(grep -vcE '^\s*(#|$)' "$HOSTS")

phase "Plan"
log "HOSTS=$HOSTS  nodes=$NHOSTS  PORT=$PORT  USER=$SSH_USER  GPU_TYPE=$GPU_TYPE  gpus/node=$GPUS_PER_NODE"
log "IMAGE=$IMAGE"
log "SHARE_DIR=$SHARE_DIR (cublas binaries on each node)"
log "Output: $OUT_DIR/percard_cluster/<node>/dev{0..$((GPUS_PER_NODE-1))}.log"
log "Transport: plain ssh (parallel). Remote runner piped via stdin — no shared FS needed for the script."

# ---------------- preflight: connectivity + image ----------------
phase "Preflight: connectivity + image"
PRE=$(_ssh_parallel_cmd \
    "echo GPUS=\$(nvidia-smi -L 2>/dev/null | grep -c '^GPU') DOCKER=\$(docker version --format '{{.Server.Version}}' 2>/dev/null||echo NA) IMG=\$(docker image inspect $IMAGE >/dev/null 2>&1 && echo HAVE || echo MISS)" \
    2>&1)
echo "$PRE" | grep -E '\[SUCCESS\]|\[FAILURE\]|GPUS=' | sed 's/^/  /'
NOK=$(echo "$PRE" | grep -c '\[SUCCESS\]')
NFAIL=$(echo "$PRE" | grep -c '\[FAILURE\]')
NMISS=$(echo "$PRE" | grep -c 'IMG=MISS')
log "Connectivity SUCCESS=$NOK FAILURE=$NFAIL / $NHOSTS ; missing image on $NMISS node(s)"
[ "$NFAIL" -eq 0 ] || die "Some nodes are unreachable. Fix connectivity before continuing."

# Pull image on nodes that are missing it
if [ "$NMISS" -gt 0 ]; then
    log "$NMISS node(s) missing image; running idempotent docker pull (only missing ones actually pull)"
    _ssh_parallel_cmd \
        "docker image inspect $IMAGE >/dev/null 2>&1 || docker pull $IMAGE" \
        2>&1 | grep -E '\[SUCCESS\]|\[FAILURE\]' | sed 's/^/  /'
fi

if [ "$PREFLIGHT_ONLY" = "1" ]; then log "--preflight: precheck complete, exiting."; exit 0; fi

if [ "$FORCE" != "1" ]; then
    read -r -p $'\033[1;33mAbout to launch per-card cuBLAS on '"$NHOSTS"$' node(s) (container per node, '"$GPUS_PER_NODE"$' cards each). Continue? [y/N] \033[0m' ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }
fi

# ---------------- parallel per-card run ----------------
phase "Parallel per-card (each node: $GPUS_PER_NODE cards, serial within node; nodes parallel)"
rm -rf "$OUT_DIR/percard_cluster"; mkdir -p "$OUT_DIR/percard_cluster"

ENV_PREFIX="SHARE_DIR='$SHARE_DIR' IMAGE='$IMAGE' GPU_TYPE='$GPU_TYPE' GPUS_PER_NODE='$GPUS_PER_NODE'"
OUT=$(_ssh_parallel_pipe "$SCRIPT_DIR/_percard_remote.sh" "$ENV_PREFIX" 2>&1)
echo "$OUT" | grep -E '\[SUCCESS\]|\[FAILURE\]|LOCALIP=|DONE=|REMOTE_ERR=' | sed 's/^/  /'
NDONE=$(echo "$OUT" | grep -c 'DONE=')
log "Completed DONE=$NDONE / $NHOSTS"

# ---------------- collect results ----------------
phase "Collect results"
# The remote runner writes logs to $SHARE_DIR/percard_cluster/<node_id>/ on each
# node. If SHARE_DIR is a shared filesystem visible from the controller, results
# are already at $SCRIPT_DIR/percard_cluster/. If not, pull them via rsync.
# We always rsync (idempotent — shared FS just copies in place).
NEED_PULL=0
while read -r h _; do
    case "$h" in ''|\#*) continue;; esac
    # Check if results are already local (shared FS case)
    if [ -d "$OUT_DIR/percard_cluster/$h" ]; then
        continue
    fi
    # Need to pull from this node
    rsync -az -e "ssh $SSH_OPTS -p $PORT" \
        "${SSH_USER}@${h}:$SHARE_DIR/percard_cluster/" \
        "$OUT_DIR/percard_cluster/" 2>/dev/null
    if [ $? -ne 0 ]; then
        log "Warning: rsync failed for $h (results may be incomplete)"
    fi
    NEED_PULL=1
done < <(grep -vE '^[[:space:]]*(#|$)' "$HOSTS")

if [ "$NEED_PULL" = "1" ]; then
    log "Results pulled from remote nodes to $OUT_DIR/percard_cluster/"
fi

# ---------------- aggregate + outlier detection ----------------
phase "Aggregate + fleet-wide outlier detection"
python3 parse_cluster_percard.py "$OUT_DIR/percard_cluster" "$GPU_TYPE"
