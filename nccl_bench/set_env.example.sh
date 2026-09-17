# ============================================================================
# set_env.example.sh — copy to set_env.sh and edit for YOUR cluster.
#
#   cp set_env.example.sh set_env.sh
#   vim set_env.sh
#
# set_env.sh is git-ignored so your site-specific values are not committed.
# `source set_env.sh` from the shell before running the NCCL scripts.
# ============================================================================

# OPTIONAL: a directory that exists at the same path on every node (a shared
# filesystem), if you want the container to write logs somewhere durable.
# Leave it empty if you do not have one — the tests do not require it, and
# launch_container.sh only bind-mounts it when it is set.
# export PREFIX_SHARE_PATH="/mnt/data"

# ---------------- bootstrap / TCP interface ----------------
# The interface mpirun and NCCL use for the control plane. Must exist with the
# same name on every node. Find it with:  ip -o -4 route show default
# On bonded fabrics use the bond device (e.g. bond0 / bond1), not a slave port.
export NETWORK_IF="${NETWORK_IF:-bond0}"

# ---------------- GPU model ----------------
# Used by split_ip.sh to decide the node-list split strategy:
#   H200 / B200 / B300 : 8 GPUs/node, sequential split
#   GB200 / GB300      : 4 GPUs/tray, 18 trays per rack but only the first 16
#                        used, so the last 2 are skipped to stay power-of-two
export GPU_TYPE="${GPU_TYPE:-H200}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# ---------------- container image ----------------
# Image used for the NCCL tests. Build it from ./Dockerfile, or pull/point at
# any CUDA image that already has nccl-tests built with the _mpi binaries.
export IMAGE_NAME="${IMAGE_NAME:-}"

# ---------------- RDMA ----------------
# Usually leave this empty: run_nccltest.sh derives NCCL_IB_HCA from GPU-NIC
# affinity via gpu_nic_topo.sh. Set it only to pin a specific rail set.
# export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3"
#
# RoCE v2 GID index. Leave empty to auto-detect; find it on a node with:
#   bash perftest/diag_gid.sh <dev>
# InfiniBand fabrics do not use this. A wrong index makes every QP fail to
# connect with "ibv_modify_qp INIT->RTR ETIMEDOUT".
# export NCCL_IB_GID_INDEX=3
