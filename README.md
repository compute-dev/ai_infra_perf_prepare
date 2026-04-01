# Environment Prerequisites

Basic Configuration (Reference)

| Item                         | Configuration                                          |
|------------------------------|--------------------------------------------------------|
| OS                           | Ubuntu 24.04.3 LTS                                    |
| driver version               | `>=570`                                                |
| cuda version                 | `>=12.9`                                               |
| docker version               | `>=26  `                                               |
| nvidia-container-cli version | 1.18.0                                                 |
| root account                 | Configure passwordless SSH login between machines using root account |
| Shared storage               | Configure NFS or use shared storage, at least 5TB      |
| Local NVMe /data disk        | Local disk must be mounted at /data, capacity > 3TB    |
| miniforge, pssh, ansible     | See install_env.sh                                     |

# Docker Image

Images

| GPU Model       | Image Address                                                                             |
|-----------------|-------------------------------------------------------------------------------------------|
| A800            | rozinnnn/ai_infra_bench:pytorch25.05-py3-te2.3-mcore0.12.1-nccl2.26.5-cuda12.9-a800-x86  |
| H200/H800       | rozinnnn/ai_infra_bench:pytorch25.05-py3-te2.3-mcore0.12.1-nccl2.26.5-cuda12.9-h200-x86  |
| B200/B300       | rozinnnn/ai_infra_bench:pytorch25.05-py3-te2.3-mcore0.13.1-nccl2.26.5-cuda12.9-b200-x86  |
| GB200/GB300     | rozinnnn/ai_infra_bench:pytorch25.05-py3-te2.3-mcore0.12.1-nccl2.26.5-cuda12.9-gb300-arch|

If you cannot pull the image, build it locally. Each machine must have the same image.

Dockerfile

```Dockerfile
FROM nvcr.io/nvidia/pytorch:25.05-py3
  
RUN apt install -y vim git wget curl

RUN cd /root && git clone https://github.com/NVIDIA/TransformerEngine.git && \
    cd /root/TransformerEngine && git checkout v2.3 && git submodule update --init --recursive && \
    NVTE_FRAMEWORK=pytorch MAX_JOBS=64 NVTE_BUILD_THREADS_PER_JOB=32 pip install .

RUN cd /root && git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM && \
    git checkout core_r0.13.0 && pip3 install . && \
    pip3 install transformers sentencepiece
```

Build image

```shell
docker build -t pytorch:25.05-py3-te2.3-mcore0.12.1 --network=host .
```

# cublasMatmulbench

```shell
cd cublas_bench
# Example
bash bash cublas_bench_b2_3.sh 2>&1 | tee cublas_bench.log | python3 parse_cublas_simple.py
```

# NCCL Test

Reference: https://github.com/NVIDIA/nccl-tests

### Single Node

```shell
git clone https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests
make
```

vim 1node_nccl.sh

```shell
echo "all_reduce_perf"
./build/all_reduce_perf -b 8 -e 16G -f 2 -g 8 -n 50

echo "alltoall_perf"
./build/alltoall_perf -b 8 -e 16G -f 2 -g 8 -n 50

echo "all_gather_perf"
./build/all_gather_perf -b 8 -e 16G -f 2 -g 8 -n 50

echo "reduce_scatter_perf"
./build/reduce_scatter_perf -b 8 -e 16G -f 2 -g 8 -n 50
```

Parameter -g: For a single machine with x GPUs, set to x

### Multi-Node
Note: ACS must be disabled

Execute on the host machine. Root accounts on all nodes must have passwordless SSH access to each other.

```shell
# Configure passwordless SSH
ssh-keygen -t rsa
# Copy public keys from all nodes to the master's authorized_keys, then distribute authorized_keys to all nodes
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
vim ~/.ssh/authorized_keys

# Verify
ssh xxxip
```

Check nvcc, mpi, nccl

```shell
which nvcc
which mpirun
ls /usr/local/cuda/include/cuda.h
ls /usr/include/mpi.h
ls /usr/include/nccl.h
# MPI is commonly missing. The following install command places MPI at /usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h by default
apt update && apt install -y openmpi-bin openmpi-common libopenmpi-dev

make MPI=1 NAME_SUFFIX=_mpi MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr
```

vim multinode_nccl.sh

```shell
echo "=============all_reduce_perf_mpi============="
/usr/local/openmpi/bin/mpirun -np 16 \
-H x.x.x.x:8,y.y.y.y:8 \
--allow-run-as-root \
--mca oob_tcp_if_include bond0 \
--mca btl_tcp_if_include bond0 \
-x NCCL_SOCKET_IFNAME=bond0 \
-x UCX_NET_DEVICES=bond0 \
-x NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
-x PATH=/usr/local/openmpi/bin:$PATH \
-x LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH \
--mca plm_rsh_args "-p 22" \
./build/all_reduce_perf_mpi -b 8 -e 16G -f 2 -g 1 -n 50

echo "=============alltoall_perf_mpi============="
/usr/local/openmpi/bin/mpirun -np 16 \
-H x.x.x.x:8,y.y.y.y:8 \
--allow-run-as-root \
--mca oob_tcp_if_include bond0 \
--mca btl_tcp_if_include bond0 \
-x NCCL_SOCKET_IFNAME=bond0 \
-x UCX_NET_DEVICES=bond0 \
-x NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
-x PATH=/usr/local/openmpi/bin:$PATH \
-x LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH \
--mca plm_rsh_args "-p 22" \
./build/alltoall_perf_mpi -b 8 -e 16G -f 2 -g 1 -n 50

echo "=============all_gather_perf_mpi============="
/usr/local/openmpi/bin/mpirun -np 16 \
-H x.x.x.x:8,y.y.y.y:8 \
--allow-run-as-root \
--mca oob_tcp_if_include bond0 \
--mca btl_tcp_if_include bond0 \
-x NCCL_SOCKET_IFNAME=bond0 \
-x UCX_NET_DEVICES=bond0 \
-x NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
-x PATH=/usr/local/openmpi/bin:$PATH \
-x LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH \
--mca plm_rsh_args "-p 22" \
./build/all_gather_perf_mpi -b 8 -e 16G -f 2 -g 1 -n 50

echo "=============reduce_scatter_perf_mpi============="
/usr/local/openmpi/bin/mpirun -np 16 \
-H x.x.x.x:8,y.y.y.y:8 \
--allow-run-as-root \
--mca oob_tcp_if_include bond0 \
--mca btl_tcp_if_include bond0 \
-x NCCL_SOCKET_IFNAME=bond0 \
-x UCX_NET_DEVICES=bond0 \
-x NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
-x PATH=/usr/local/openmpi/bin:$PATH \
-x LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH \
--mca plm_rsh_args "-p 22" \
./build/reduce_scatter_perf_mpi -b 8 -e 16G -f 2 -g 1 -n 50
```

# Preparation Acceptance Checklist

### Environment Configuration

Based on actual installation

| Item                         | Completed | Actual Completed Version |
|------------------------------|-----------|--------------------------|
| OS                           | ✅        | Ubuntu 24.04.3 LTS      |
| driver version               | ✅        | `>=570`                  |
| cuda version                 |           | `>=12.9`                 |
| docker version               |           | `>=26  `                 |
| nvidia-container-cli version |           | 1.18.0                   |
| root account                 |           |                          |
| Shared storage               |           | 5TB                      |
| Local NVMe /data disk        |           | /data, 3TB               |
| miniforge                    |           |                          |
| pssh                         |           |                          |
| ansible                      |           |                          |

### cuBLAS Bench Results

Submit log files

### NCCL Test Results

packet=16G

| Vendor | GPU*GPUs_per_node*Nodes | AllReduce busbw(GB/s) | All2All busbw(GB/s) | AllGather busbw(GB/s) | ReduceScatter busbw(GB/s) | NIC |
|:------:|:-----------------------:|:---------------------:|:-------------------:|:---------------------:|:-------------------------:|:---:|
|  xxx   |       xxx\*8\*1         |          xxx          |         xxx         |          xxx          |            xxx            | CX? |
|  xxx   |       xxx\*8\*2         |          ...          |         ...         |          ...          |            ...            | ... |
|  xxx   |       xxx\*8\*4         |          xxx          |         xxx         |          xxx          |            xxx            | CX? |
|  ...   |          ...            |          ...          |         ...         |          ...          |            ...            | ... |
