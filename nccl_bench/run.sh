#!/bin/bash
# ============================================================================
# run.sh — container entrypoint (image CMD).
#
# Writes an ssh config that lets every node reach every other node as root
# without a password, starts sshd on $PORT, then drops to an interactive shell.
#
# The image ships one keypair, generated at build time, so all containers from
# the same image trust each other. That is what makes `mpirun -H` work.
# ============================================================================
set -x

port="${PORT:-7900}"

# Node list: HOSTS_FILE is injected by launch_container.sh and points into the
# mounted directory; fall back to /root/hosts.txt.
HOSTS_FILE="${HOSTS_FILE:-/root/hosts.txt}"
if [ -z "${allips}" ] && [ -f "${HOSTS_FILE}" ]; then
    allips=$(grep -vE '^\s*(#|$)' "${HOSTS_FILE}" | awk '{print $1}' | paste -sd ',' -)
    export allips
    echo "[INFO] Node list read from ${HOSTS_FILE}: ${allips}"
fi

# Point the ssh client at the port sshd listens on
if [ -f /root/.ssh/config ]; then
    sed -i "s/^Port .*/Port $port/" /root/.ssh/config
else
    echo "Port $port" > /root/.ssh/config
    echo "StrictHostKeyChecking no" >> /root/.ssh/config
fi

# This container's own bootstrap address
if [ -n "${NCCL_SOCKET_IFNAME}" ]; then
    echo "[INFO] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
    ip=$(ip -4 -o addr show "${NCCL_SOCKET_IFNAME}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
    if [ -n "${ip}" ]; then
        {
            echo "Host $ip"
            echo "  StrictHostKeyChecking no"
        } >> /root/.ssh/config
    fi
else
    echo "[WARN] NCCL_SOCKET_IFNAME not set — set it so mpirun binds the right interface"
fi

# Pre-trust every peer
echo "${allips}"
IFS=',' read -ra ips <<< "$allips"
for ip in "${ips[@]}"; do
    {
        echo "Host $ip"
        echo "  StrictHostKeyChecking no"
    } >> /root/.ssh/config
done

/usr/sbin/sshd -p"$port" -o ListenAddress=0.0.0.0

bash
