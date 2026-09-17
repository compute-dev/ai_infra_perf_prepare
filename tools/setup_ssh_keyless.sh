#!/bin/bash
# ============================================================================
# setup_ssh_keyless.sh — set up passwordless ssh from this machine to the nodes.
#
# This is the first thing to run on a new cluster. It pushes this machine's
# public key into each node's authorized_keys and then verifies that passwordless
# login actually works.
#
# The key is APPENDED AND DE-DUPLICATED, never overwritten — an existing
# authorized_keys on the node keeps its other keys, so you cannot lock anyone
# out by re-running this.
#
# Usage:
#     bash setup_ssh_keyless.sh <hostfile|ip1,ip2,...> [options]
#
# Options:
#     --port N            ssh port (default 22)
#     --user USER         remote user (default root)
#     --password PASS     node password. Prefer the NODE_PASSWORD env var or the
#                         interactive prompt: an argument is visible in `ps`.
#
# Examples:
#     NODE_PASSWORD='...' bash setup_ssh_keyless.sh hosts.txt
#     bash setup_ssh_keyless.sh 10.0.0.1,10.0.0.2 --user ubuntu --port 2222
#
# Requires: sshpass (for the initial password-authenticated push), ssh-keygen.
# ============================================================================
set -uo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

TARGETS="$1"; shift
PORT="${SSH_PORT:-22}"
USER_NAME="${SSH_USER:-root}"
PASSWD="${NODE_PASSWORD:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --port)      PORT="${2:-}"; shift 2 ;;
        --user)      USER_NAME="${2:-}"; shift 2 ;;
        --password)  PASSWD="${2:-}"; shift 2 ;;
        -h|--help)   sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown option: $1 (use --help)" >&2; exit 1 ;;
    esac
done

command -v sshpass >/dev/null 2>&1 || {
    echo "[ERROR] sshpass is required for the initial key push." >&2
    echo "        Install it (apt install sshpass / yum install sshpass) or add the" >&2
    echo "        key to the nodes by hand, then re-run just the verification." >&2
    exit 1
}

# ---- expand the target list ------------------------------------------------
IPS=()
if [ -f "${TARGETS}" ]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "${line}" | xargs)"
        [ -n "${line}" ] && IPS+=("${line}")
    done < "${TARGETS}"
else
    IFS=',' read -ra IPS <<< "${TARGETS}"
fi
[ "${#IPS[@]}" -gt 0 ] || { echo "[ERROR] no usable targets" >&2; exit 1; }

# ---- password --------------------------------------------------------------
if [ -z "${PASSWD}" ]; then
    read -r -s -p "Password for ${USER_NAME}@<nodes> (not stored): " PASSWD
    echo
fi
[ -n "${PASSWD}" ] || { echo "[ERROR] no password given" >&2; exit 1; }

# ---- local key -------------------------------------------------------------
# Existing key is reused, never regenerated
[ -f ~/.ssh/id_ed25519.pub ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
PUBKEY="$(cat ~/.ssh/id_ed25519.pub)"
[ -n "${PUBKEY}" ] || { echo "[ERROR] public key is empty" >&2; exit 1; }

echo "===== keyless ssh setup -> ${#IPS[@]} node(s), port ${PORT}, user ${USER_NAME} ====="
echo "public key fingerprint: $(ssh-keygen -lf ~/.ssh/id_ed25519.pub 2>/dev/null | awk '{print $2}')"

COMMON="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

OK=0; FAIL=0
for ip in "${IPS[@]}"; do
    ip="$(echo "${ip}" | xargs)"
    [ -z "${ip}" ] && continue

    if sshpass -p "${PASSWD}" ssh -p "${PORT}" ${COMMON} -o ConnectTimeout=12 "${USER_NAME}@${ip}" \
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && { grep -qxF '${PUBKEY}' ~/.ssh/authorized_keys 2>/dev/null || echo '${PUBKEY}' >> ~/.ssh/authorized_keys; } && chmod 600 ~/.ssh/authorized_keys" 2>/dev/null; then

        # Verify it really is keyless — not a false OK from a cached agent
        if ssh -p "${PORT}" ${COMMON} -o ConnectTimeout=8 -o BatchMode=yes "${USER_NAME}@${ip}" 'exit 0' 2>/dev/null; then
            echo "[ OK ] ${ip}"
            OK=$((OK + 1))
        else
            echo "[FAIL] ${ip}  key was pushed but keyless login still fails (does sshd allow pubkey auth?)"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "[FAIL] ${ip}  push failed (wrong password, wrong port, or unreachable)"
        FAIL=$((FAIL + 1))
    fi
done

echo "===== result: OK=${OK} FAIL=${FAIL} / ${#IPS[@]} ====="
[ "${FAIL}" -eq 0 ] || exit 1
