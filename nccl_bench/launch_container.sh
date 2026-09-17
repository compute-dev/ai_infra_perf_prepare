#!/bin/bash
# ============================================================================
# launch_container.sh — start the `nccltest` container on THIS host.
#
# Run it on every node of the cluster (same command everywhere). It starts a
# detached, privileged container that:
#   - sees all GPUs and the host network / RDMA devices
#   - runs sshd on $PORT so mpirun can reach peer nodes
#   - bind-mounts the directory holding your host list, so editing that file on
#     the host takes effect in the container immediately, with no restart
#
# Usage (on each node):
#     IMAGE_NAME=<your image> \
#     NCCL_SOCKET_IFNAME=<iface> \
#     HOSTS_FILE=/path/to/hosts.txt \
#     bash launch_container.sh
#
# Required env:
#     IMAGE_NAME            container image that has nccl-tests built (see Dockerfile)
#     HOSTS_FILE            node list file, one IP per line  (or set allips instead)
# Optional env:
#     NCCL_SOCKET_IFNAME    bootstrap/TCP interface; auto-detected in-container if unset
#     PORT                  in-container sshd port (default 7900 — do not change
#                           unless you also change it in run_nccltest.sh)
#     PREFIX_SHARE_PATH     extra host directory to bind-mount at the same path
#                           (only needed if you want logs on shared storage)
#     SHM_SIZE              /dev/shm size (default 400gb)
# ============================================================================
set -x

name=nccltest

# --- image (required: no vendor default is baked in) ------------------------
if [ -z "${IMAGE_NAME}" ]; then
    echo "[ERROR] IMAGE_NAME is not set." >&2
    echo "        Build one from ./Dockerfile, or pull a CUDA image that already" >&2
    echo "        contains nccl-tests with the _mpi binaries." >&2
    echo "        Example: IMAGE_NAME=<registry>/<image>:<tag> bash $0" >&2
    exit 1
fi
image_name="${IMAGE_NAME}"

SSH_PORT="${PORT:-7900}"

# --- node list (host list file preferred over the allips env var) -----------
HOSTS_FILE="${HOSTS_FILE:-$(cd "$(dirname "$0")" && pwd)/hosts.txt}"
allips="${allips:-}"

if [ ! -f "${HOSTS_FILE}" ] && [ -z "${allips}" ]; then
    echo "[ERROR] No node list. Do one of:" >&2
    echo "  1) create hosts.txt next to this script, one IP per line (recommended:" >&2
    echo "     it can be edited later without restarting the container)" >&2
    echo "  2) pass it in: allips=ip1,ip2 bash $0" >&2
    exit 1
fi

if [ -f "${HOSTS_FILE}" ]; then
    NODE_NUM=$(grep -vE '^\s*(#|$)' "${HOSTS_FILE}" | wc -l)
    echo "[INFO] Found ${HOSTS_FILE} (${NODE_NUM} nodes) — will be mounted into the container"
    echo "[NOTE] Edit that file on the host at any time to change the node list; no restart needed"
fi

docker stop $name 2>/dev/null
docker rm $name 2>/dev/null

EXTRA_ARGS=()
if [ -f "${HOSTS_FILE}" ]; then
    # File mode: mount the DIRECTORY, not the file. Editors such as vim replace
    # the inode on save, and a bind-mounted single file then goes stale inside
    # the container ("Stale file handle"). Mounting the directory avoids it.
    # Deliberately do NOT also pass -e allips here: an environment variable
    # would freeze the list and shadow the live file.
    HOSTS_DIR="$(cd "$(dirname "${HOSTS_FILE}")" && pwd)"
    HOSTS_BASENAME="$(basename "${HOSTS_FILE}")"
    EXTRA_ARGS+=(-v "${HOSTS_DIR}:/root/hosts_dir")
    EXTRA_ARGS+=(-e "HOSTS_FILE=/root/hosts_dir/${HOSTS_BASENAME}")
    if [ -n "${PREFIX_SHARE_PATH}" ]; then
        EXTRA_ARGS+=(-v "${PREFIX_SHARE_PATH}:${PREFIX_SHARE_PATH}")
    fi
else
    # Legacy path: no host list file, so the env var is the only source (and a
    # container restart is needed to change it)
    EXTRA_ARGS+=(-e "allips=${allips}")
fi

docker run -it -d --name ${name} --gpus all --ipc=host --network=host --privileged --device=/dev/infiniband --ulimit=memlock=-1 \
      -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}" \
      -e PORT="${SSH_PORT}" \
      "${EXTRA_ARGS[@]}" \
      --shm-size="${SHM_SIZE:-400gb}" ${image_name}
