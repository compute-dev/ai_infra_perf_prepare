# ============================================================================
# nccltest image — CUDA runtime + nccl-tests (with MPI binaries) + sshd.
#
# Build:
#     docker build -t nccltest:latest .
#     docker build --build-arg BASE=nvcr.io/nvidia/pytorch:25.05-py3 -t nccltest:cu12 .
#
# The build needs internet access to the base image registry, to
# apt repositories, and to github.com for nccl-tests. If your cluster is
# air-gapped, build the image elsewhere and `docker load` it on each node.
#
# Multi-arch, if you need x86_64 and arm64 from one build:
#     docker run --privileged --rm tonistiigi/binfmt --install all
#     docker buildx create --use --name multiarch
#     docker buildx build --platform linux/amd64,linux/arm64 -t <registry>/nccltest:tag --push .
# ============================================================================

ARG BASE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${BASE}

WORKDIR /root

# Base tooling: sshd (so mpirun can reach peer nodes), plus network and PCI
# utilities the test scripts and diagnostics rely on.
RUN apt-get clean all && apt-get update \
    && apt-get remove -y openssh-client && apt-get install -y openssh-client \
    && apt-get install -y openssh-server \
    && apt-get install -y --no-install-recommends \
       net-tools \
       iproute2 \
       iputils-ping \
       curl \
       wget \
       vim \
       dnsutils \
       telnet \
       lsof \
       htop \
       pciutils \
       ethtool \
       numactl \
    && rm -rf /var/lib/apt/lists/*

# nccl-tests. Plain `make` also builds the single-process binaries that
# 1node_nccltest.sh uses; the _mpi suffix ones are what run_nccltest.sh uses.
RUN git clone https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests \
    && make clean && make \
    && make MPI=1 NAME_SUFFIX=_mpi MPI_HOME=/usr/local/mpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr

# sshd host keys
RUN ssh-keygen -A && mkdir -p /run/sshd

# Passwordless root ssh BETWEEN containers of this image.
#
# One keypair is generated at build time and baked in, so every container from
# the same image trusts every other. That is what makes `mpirun -H ip1,ip2`
# work without any per-run key distribution. Note this key is per-image, not a
# shared secret shipped in the repo, and the container is meant to run on a
# trusted, isolated fabric. Rebuild the image to rotate it.
RUN mkdir -p /root/.ssh && \
    ssh-keygen -t ed25519 -N '' -f /root/.ssh/id_rsa && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/id_rsa /root/.ssh/authorized_keys && \
    echo "Port ${PORT:-7900}" > /root/.ssh/config && \
    echo "StrictHostKeyChecking no" >> /root/.ssh/config && \
    echo "UserKnownHostsFile /dev/null" >> /root/.ssh/config && \
    chmod 600 /root/.ssh/config

COPY run.sh /root/run.sh
COPY run_nccltest.sh /root/run_nccltest.sh
COPY 1node_nccltest.sh /root/1node_nccltest.sh
COPY parse_nccl_log.py /root/parse_nccl_log.py
COPY gpu_nic_topo.sh /root/gpu_nic_topo.sh
COPY launch_container.sh /root/launch_container.sh

CMD ["bash", "/root/run.sh"]
