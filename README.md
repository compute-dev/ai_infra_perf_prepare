# 环境预装

基础配置（参考）

| 项目                           | 配置说明                  |
|------------------------------|-----------------------|
| OS                           | Ubuntu 24.04.3 LTS    |
| driver version               | `>=570`               |
| cuda version                 | `>=12.9`              |
| docker version               | `>=26  `              |
| nvidia-container-cli version | 1.18.0                |
| root账号                       | 基于root账号，配置机器间ssh免密登录 |
| 共享存储                         | 配置nfs，或者使用共享存储，至少5TB  |
| 本地nvme /data盘                | 本地盘要挂载上/data，容量大于3TB  |
| miniforge、pssh、ansible       | 见 install_env.sh      |

# Docker 镜像

镜像

| 显卡型号        | 镜像地址                                                                                      |
|-------------|-------------------------------------------------------------------------------------------|
| A800        | rozinnnn/ai_infra_bench:pytorch25.05-py3-te2.3-mcore0.12.1-nccl2.26.5-cuda12.9-a800-x86   |
| H200/H800   | rozinnnn/ai_infra_bench:pytorch25.05-py3-te2.3-mcore0.12.1-nccl2.26.5-cuda12.9-h200-x86   |
| B200/B300   | rozinnnn/ai_infra_bench:pytorch25.05-py3-te2.3-mcore0.13.1-nccl2.26.5-cuda12.9-b200-x86   |
| GB200/GB300 | rozinnnn/ai_infra_bench:pytorch25.05-py3-te2.3-mcore0.12.1-nccl2.26.5-cuda12.9-gb300-arch |

如果不能拉取，则本地build，每台机器都需要相同的镜像

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

build 镜像

```shell
docker build -t pytorch:25.05-py3-te2.3-mcore0.12.1 --network=host .
```

# cublasMatmulbench

```shell
cd cublas_bench
# 示例
bash bash cublas_bench_b2_3.sh 2>&1 | tee cublas_bench.log | python3 parse_cublas_simple.py
```

# NCCL test

参考https://github.com/NVIDIA/nccl-tests

### 单机

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

参数 -g：单机 x 卡，则 x

### 多机
注：需要关闭 ACS

宿主机执行，宿主机之间各节点机器 root 账户互相免密

```shell
# 配置免密
ssh-keygen -t rsa
# 将其他节点的公钥都拷贝到master的authorized_keys中，然后所有的节点都拷贝一份authorized_keys过去
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
vim ~/.ssh/authorized_keys

# 验证
ssh xxxip
```

检查 nvcc，mpi，nccl

```shell
which nvcc
which mpirun
ls /usr/local/cuda/include/cuda.h
ls /usr/include/mpi.h
ls /usr/include/nccl.h
# 一般可能缺少mpi，下面安装命令mpi默认路径为/usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h
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

# prepare验收单

### 环境配置

以实际安装为准

| 项目                           | 是否完成 | 实际已完成版本号           |
|------------------------------|------|--------------------|
| OS                           | ✅    | Ubuntu 24.04.3 LTS |
| driver version               | ✅    | `>=570`            |
| cuda version                 |      | `>=12.9`           |
| docker version               |      | `>=26  `           |
| nvidia-container-cli version |      | 1.18.0             |
| root账号                       |      |                    |
| 共享存储                         |      | 5TB                |
| 本地nvme /data盘                |      | /data，3TB          |
| miniforge                    |      |                    |
| pssh                         |      |                    |
| ansible                      |      |                    |

### cublasbench结果

提交日志文件

### nccltest结果

packet=16G

| 厂商  | 卡型\*单机卡数\*机器数 | AllReduce busbw(GB/s) | All2All busbw(GB/s) | AllGather busbw(GB/s) | ReduceScatter busbw(GB/s) | 网卡  |
|:---:|:-------------:|:---------------------:|:-------------------:|:---------------------:|:-------------------------:|:---:|
| xxx |   xxx\*8\*1   |          xxx          |         xxx         |          xxx          |            xxx            | CX? |
| xxx |   xxx\*8\*2   |          ...          |         ...         |          ...          |            ...            | ... |
| xxx |   xxx\*8\*4   |          xxx          |         xxx         |          xxx          |            xxx            | CX? |
| ... |      ...      |          ...          |         ...         |          ...          |            ...            | ... |
