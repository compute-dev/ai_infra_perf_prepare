**中文** | [English](readme.md)

# cuBLAS GEMM Benchmark

面向 NVIDIA GPU（B200、B300、GB200、GB300、H100、H200、H800、A800、H20、L20）的
单卡 GEMM TFLOPS 基准测试。在所有支持的精度（FP4 / FP8 / INT8 / FP16 / BF16 /
TF32 / FP32）上测量每卡算力 throughput，并与官方 baseline 对比。

## 本工具测量什么

- **仅单卡 GEMM TFLOPS。** 每张卡通过 `cublasMatmulBench` 独立测试，这是一个
  cuBLASLt 矩阵乘 kernel，采用 peak-saturating shapes。结果是每张卡、每个精度
  的一个 TFLOPS 数值。

## 本工具不测量什么

- **NVLink、RDMA 和拓扑。** 本工具只测算力。如需互连带宽与拓扑验证，请使用
  `../nccl_bench/`。
- **多卡集合通信性能。** 这里没有 NCCL 或多卡测试。

## 前置条件

### Docker image

`cublasMatmulBench_<GPU>` 二进制文件**随本 repo 提供**，**不在** image 中。image
只提供 CUDA 运行时库。

| GPU family           | CUDA era  | Image example                         |
|----------------------|-----------|---------------------------------------|
| Blackwell (B200/B300)| CUDA 13+  | `nvcr.io/nvidia/pytorch:25.10-py3`    |
| Grace-Blackwell (GB200/GB300) | CUDA 13+ | `nvcr.io/nvidia/pytorch:25.10-py3` |
| Hopper (H100/H200/H800/A800) | CUDA 12 | `nvcr.io/nvidia/pytorch:25.05-py3` |
| Ampere (H20/L20)     | CUDA 12   | `nvcr.io/nvidia/pytorch:25.05-py3`    |

> 具体的 image 不重要，只要它带有匹配的 CUDA 运行时即可。Blackwell 和
> Grace-Blackwell 部件需要 CUDA 13+ 时代的 image；Hopper 和 Ampere 可使用
> CUDA 12 时代的 image。

### 主机要求

- 已安装 NVIDIA driver（`nvidia-smi` 可用）
- 带 GPU 支持的 Docker（`--gpus all`）
- 每个节点上可访问 `cublas_bench/` 目录（共享文件系统或本地拷贝）

## 快速开始：单节点测试

```bash
cd /path/to/cublas_bench

# 通过 docker 包装脚本(推荐 —— 会处理 CUDA 库依赖)
GPU_NAME=H800 image=nvcr.io/nvidia/pytorch:25.05-py3 bash auto_cublas_bench_by_docker.sh

# 直接运行(宿主机上已有 CUDA 运行时)
GPU_NAME=H800 bash auto_cublas_bench.sh
```

输出会 tee 到 `cublas_bench_<GPU>.log`，并实时对照
`cublasMatmulBenchBaseline.yaml` 解析。退出码 0 = 全部通过，1 = 部分失败，
2 = 用法错误，3 = 本次运行未产出可用结果（缺少 CUDA 库，或日志里没有任何可解析
内容）。

### 只测 0 号卡的坑

快速测试**只跑 GPU 0**。`docker run --gpus all` 只是把所有卡暴露给容器；
`cublasMatmulBench` 没有 `--device` 参数，因此它总是在 device 0 上运行（受
`CUDA_VISIBLE_DEVICES` 约束）。

**"这台节点通过了" 并不意味着全部 8 张卡都通过了。** 真正的验收请使用下面的全量
集群逐卡扫描。

## 全量集群逐卡扫描

### 概览

集群扫描在**每个节点的每张 GPU** 上运行 `cublasMatmulBench`，收集结果，并进行
全集群 outlier 检测。这才是验收一个集群的正确方式：你要找的是**一张在多个精度
上都持续偏低的卡**（cross-fleet outlier），而不是某个精度恰好略低于 baseline。

### 涉及的三个脚本

| Script                   | Role                                                        |
|--------------------------|-------------------------------------------------------------|
| `cluster_cublas_percard.sh` | 编排器：SSH 到每个节点，管道运行 runner，收集结果       |
| `_percard_remote.sh`     | 远端 runner：单节点上的逐卡循环（通过 stdin 管道传入） |
| `parse_cluster_percard.py`| 聚合器：集群中位数、outlier 检测、CSV 输出   |

### 用法

```bash
# 1. 创建 hosts 文件(每行一个主机,允许 # 注释)
cat > hosts.txt <<'EOF'
node01
node02
node03
EOF

# 2. 预检:检查连通性 + 镜像是否存在
bash cluster_cublas_percard.sh -H hosts.txt -i nvcr.io/nvidia/pytorch:25.05-py3 --preflight

# 3. 全量扫描(交互式 —— 会要求确认)
bash cluster_cublas_percard.sh -H hosts.txt -i nvcr.io/nvidia/pytorch:25.05-py3 -g H800

# 4. 非交互
bash cluster_cublas_percard.sh -H hosts.txt -i nvcr.io/nvidia/pytorch:25.05-py3 -g H800 -y
```

### 集群脚本选项

| Flag           | Default     | Description                                            |
|----------------|-------------|--------------------------------------------------------|
| `-H <file>`    | (required)  | 主机列表文件                                         |
| `-i <image>`   | (required)  | Docker image                                           |
| `-P <port>`    | 22          | SSH 端口                                              |
| `-u <user>`    | root        | SSH 用户                                              |
| `-g <gpu>`     | H800        | GPU 类型                                              |
| `-s <dir>`     | $PWD        | 远端节点上 cublas_bench 的路径                       |
| `-n <n>`       | 8           | 每节点 GPU 数（若 _percard_remote 在没有它的情况下运行则自动检测） |
| `-o <dir>`     | `<repo>/results/cublas` | 逐卡日志落在**本机**的位置 |
| `--preflight`  |             | 只做连通性 + image 预检                              |
| `-y`           |             | 非交互式                                              |

> `-H` 和 `-o` 相对于**你调用脚本的位置**解析，而不是相对于
> `cublas_bench/`，因此从 repo 根目录用 `-H preflight/hosts.txt` 是可以的。
> `-s` 故意*不在本地解析*——它是**节点上**的路径。

### 输出

逐卡日志落在 `results/cublas/percard_cluster/<node>/dev<N>.log`，每张 GPU 一个
文件，外加 `fleet_percard.csv`。该目录正是
`tools/collect_results.sh` 打包的目录，因此无需手动搬移文件。

```bash
bash cluster_cublas_percard.sh -H ../preflight/hosts.txt -P 22 -u root \
     -i <image> -g H800 -s /data/ai_infra_perf_prepare/cublas_bench
```

单卡运行时间约 5 分钟，因此一台 8-GPU 节点约 40 分钟。
`--preflight` 只检查连通性和 image，不运行任何东西。

### 传输

纯 SSH，通过后台 job 并行。无需 `pssh`，无需 `conda`，无需 jump host，runner
脚本也不需要共享 NFS。远端 runner `_percard_remote.sh` **通过 stdin 管道传入**：

```
ssh host 'SHARE_DIR=/path IMAGE=img GPU_TYPE=H800 bash -s' < _percard_remote.sh
```

每个节点上必须能在 `SHARE_DIR` 访问到 cublas 二进制（共享文件系统或事先拷贝
过去）。

### Outlier 检测依据

判定一张卡，关键是**跨集群一致性**：一张在多个精度上都持续偏低的卡才是弱卡。
某个精度恰好略低于 baseline **不**算失败——它可能是 shape 或算法 artifact。只有
当某张卡的一个或多个精度低于该精度的**集群中位数** 3% 以上时，聚合器才把它标记
为 outlier。

跨集群 ~5–6% 的离散是正常的。单节点内，卡间差异应在 ~3% 以内。

## 第二意见工具：torch_peak_probe.py

一个配套工具，通过 PyTorch 在真实框架路径上测量 GEMM peak。它**在运行时解析同
目录下的 `cublas_bench_<GPU>.sh`** 以获取完全相同的 m/n/k shapes——单一事实来源，
没有硬编码维度。

### 用法

```bash
# 空跑(无需 GPU/torch —— 只打印将要测试的 shape)
python3 torch_peak_probe.py --gpu H800 --dry-run

# 实际运行(在带 torch 的容器内)
CUDA_VISIBLE_DEVICES=0 python3 torch_peak_probe.py --gpu H800

# 随机输入(读数比全零输入低 ~8–12 %)
CUDA_VISIBLE_DEVICES=0 python3 torch_peak_probe.py --gpu H800 --randn
```

### 两个关键陷阱

1. **随机 vs. 零输入。** 随机输入的读数比全零低约 ~8–12%。
   与 cublas 对比**必须使用 `--zero`**（默认）以匹配 cublas 的 `-B=0` 运行。
   绝不要把 `--randn` 的 torch 数值与 `-B=0` 的 cublas 数值相比。

2. **H800 的 TF32 经由 cublas `sss_fast_tf32` 在某些板子上可能读到约一半速度。**
   这是 **cublas 工具的 artifact，不是弱卡。** 真实框架使用 `allow_tf32`
   （标准 `COMPUTE_32F_TF32`），同一块板子在那里测得全速。**绝不要仅凭 cublas
   `sss_fast_tf32` 的数值就判定一张卡失败或对板子发起 RMA**——TF32 应以 torch
   `allow_tf32` 为准。另请注意：**INT8 peak 以 cublas（`bisb_imma`）为准，不是
   torch**——torch 没有干净的 INT8 peak 入口，因此 `torch_peak_probe.py` 完全
   跳过 INT8。

## 日志噪声陷阱

在 Docker 内运行时，你可能看到类似这样的行：

```
libcompat_fix.so ... cannot be preloaded
```

这些来自主机的 `/etc/ld.so.preload` 泄漏进容器。它们与基准测试结果**无关**。
用以下命令过滤掉：

```bash
grep -vE 'libcompat_fix|ld.so' cublas_bench_H800.log
```

## 文件参考

| File                          | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `auto_cublas_bench.sh`        | 单节点入口：选择 GPU 脚本、运行、解析                   |
| `auto_cublas_bench_by_docker.sh` | 单节点入口，通过 Docker 容器                          |
| `cublas_bench_<GPU>.sh`       | 每 GPU 的基准测试脚本（shapes 和 -P codes）             |
| `cublasMatmulBench_<GPU>`     | 每个 GPU 族的预编译二进制                               |
| `cublasMatmulBenchBaseline.yaml` | 官方 TFLOPS baseline（GPU -> precision -> TFLOPS）   |
| `parse_cublas_simple.py`      | 单日志解析器：实测 vs. baseline，退出码                  |
| `cluster_cublas_percard.sh`   | 集群编排器：SSH 扫描、收集、聚合                         |
| `_percard_remote.sh`          | 远端 runner：单节点逐卡循环（stdin 管道传入）           |
| `parse_cluster_percard.py`    | 集群聚合器：中位数、outlier、CSV                           |
| `torch_peak_probe.py`         | 第二意见工具：在相同 shapes 上的 torch GEMM peak          |
| `diagnose_cuda.sh`            | CUDA 运行时诊断（用于缺库错误）                          |
| `GB300/gemm-memread/`         | 独立的基于 CUTLASS 的 GEMM+memread 包（自带文档）       |

## 故障排查

| Symptom                                              | Cause / Fix                                              |
|------------------------------------------------------|----------------------------------------------------------|
| `the input device is not a TTY`                      | Docker 包装器使用 `-i`（不是 `-it`）。若需要 TTY，设置 `DOCKER_TTY=1`。 |
| `libcublasLt.so.12: cannot open shared object file` | 容器或主机中没有 CUDA 运行时。使用合适的 CUDA image，或运行 `diagnose_cuda.sh`。 |
| `libcompat_fix.so ... cannot be preloaded`           | 主机 `/etc/ld.so.preload` 泄漏进容器。无害；用 `grep -vE 'libcompat_fix|ld.so'` 过滤。 |
| 快速测试通过但某张卡实际很慢         | 快速测试只跑 GPU 0。运行全量集群逐卡扫描。 |
| H800 TF32 读数约半速                          | cublas `sss_fast_tf32` artifact。TF32 以 torch `allow_tf32` 为准，而非 cublas。 |
| `Error: image/IMAGE is required`                     | Docker 包装器要求 `-i <image>` 或 `image=<img>` 环境变量。 |
| `Error: -H <hosts_file> is required`                 | 集群脚本要求 `-H <file>` 带主机列表。  |
| Exit code 3                                          | 本次运行未产出可用结果——要么缺少 CUDA 共享库，要么日志里没有可解析内容。**这不是通过。** |
| Exit code 1                                          | 一个或多个精度低于 baseline x 95%。        |
| `Result: NO RESULTS PARSED`                          | 基准测试从未产出计时——崩溃、容器错误，或在第一个结果前就出错。不是通过。 |

## GB300/gemm-memread

`GB300/gemm-memread/` 是一个独立的基于 CUTLASS 的 GEMM + memory-read 基准测试
包，自带文档位于 `GB300/gemm-memread/docs/README.md`。它**不属于本流程**——它是
一个独立工具，用于 GB300 专用的 memory-bandwidth-aware GEMM 测量。
