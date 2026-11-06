**中文** | [English](README.md)

# AI 基础设施性能 —— 厂商准备与验收包

本仓库是 GPU 集群 POC 的**面向厂商**交付包。它包含准备集群、产出验收测量所需的
全部内容,可由**厂商在厂商自己的集群上**运行:脚本接受你提供的主机清单、SSH 端口和
输出目录,不依赖我们任何内网资源。

按顺序执行四个关卡。每一关都产出与硬件无关、可复核、可重跑、可横向对比的证据。

## 四个关卡

| # | 关卡 | 目录 | 证明什么 | 大致耗时 |
| --- | --- | --- | --- | --- |
| 0 | **节点一致性** | [`preflight/`](preflight/readme.md) | 各节点确实同构。`driver_version` 一致**并不够** —— 这里的配置漂移会静默地让后续所有对比失效。 | 几分钟 |
| 1 | **单卡算力** | [`cublas_bench/`](cublas_bench/readme.md) | 逐卡 GEMM 算力(FP8/INT8/FP16/BF16/TF32/FP32,适用时含 FP4),覆盖**每一张卡**,并用离群检测找出弱卡或降频卡。 | ~10 分钟/集群 |
| 2 | **NCCL 互联** | [`nccl_bench/`](nccl_bench/readme.md) | 机内 NVLink 与跨机 RDMA 带宽,覆盖 AllReduce / All2All / AllGather / ReduceScatter,并在多个集群规模下测量。 | ~30 分钟 |
| 3 | **RDMA 点对点** | [`perftest/`](perftest/readme.md) | NCCL 之下的裸 fabric,逐 rail 测量。把 fabric 故障与 NCCL/配置故障区分开,并能抓到只看带宽发现不了的错误包。 | ~20 分钟 |

第 0 关必须先做:节点不同构,后面的数字就不可比。第 1、2 关是主要的验收结论。
第 3 关是定位工具,告诉你第 2 关的问题**出在哪里**。

### 推理验收 —— 可选,仅推理卡

推理卡在上面四关之后还有一关,目前是**草案**:
[`inference_bench/`](inference_bench/README.zh.md) 测固定 SLO 下的单机 Prefill-only
与 Decode-only 容量。它的范围是刻意收窄的 —— 不测端到端 PD、不测 KV 传输、不测多机 ——
并且**不在** `tools/collect_results.sh` 与 `tools/summarize.py` 的覆盖范围内,结果需手工打包。
只在第 0~3 关验收通过后再跑;弱卡或坏链路会污染它的结果。
提交内容见 [SUBMISSION.zh.md](SUBMISSION.zh.md) 第 6 节。

## 快速开始

```bash
git clone <this repo> && cd ai_infra_perf_prepare

# 0. 一次性:打通本机到所有节点的免密 ssh
cp preflight/hosts.txt.example preflight/hosts.txt && vim preflight/hosts.txt
NODE_PASSWORD='...' bash tools/setup_ssh_keyless.sh preflight/hosts.txt --port 22

# 1. 节点一致性(本机运行,脚本经 ssh 管道下发到各节点)
mkdir -p results/preflight
bash preflight/cluster_runtime_check.sh -H preflight/hosts.txt -P 22 -o results/preflight/runtime.csv
bash preflight/cluster_spec_check.sh    -H preflight/hosts.txt -P 22 -o results/preflight/spec.csv

#    ... 先把本仓库复制到各节点的 <NODE_DIR> —— 见 VENDOR_RUNBOOK.md 第 3 步

# 2. 单卡算力,逐卡,覆盖所有节点
bash cublas_bench/cluster_cublas_percard.sh \
     -H preflight/hosts.txt -P 22 -u root \
     -i <IMAGE> -g H800 -s <NODE_DIR>/cublas_bench

# 3. NCCL 互联(autoconf 需要读取节点信息,所以要在某个节点上跑)
ssh <node> "cd <NODE_DIR>/nccl_bench && bash nccltest_autoconf.sh --apply"
#    ... 然后在每个节点启动容器、从其中一个节点发起测试 —— 见 nccl_bench/readme.md

# 4. RDMA 点对点
bash perftest/rail_isolation_check.sh node01 node02   # 先做这个:它决定后面怎么分组
bash perftest/perftest_pair.sh node01 node02
bash perftest/perftest_agg.sh  node01 node02

# 5. 打包所有内容用于提交
bash tools/collect_results.sh --vendor <name> --gpu H800 --dir results
python3 tools/summarize.py results --vendor <name> --gpu H800
```

以上只是简版 —— 真正要照着做的是 **[VENDOR_RUNBOOK.md](VENDOR_RUNBOOK.md)**,
里面包含每关应该产出什么、怎么判读、以及各种坑。需要填写的验收表在
**[SUBMISSION.md](SUBMISSION.md)**。

## 前置条件

### 集群

| 项目 | 要求 |
| --- | --- |
| OS | Ubuntu 24.04 LTS(其他现代发行版亦可;脚本是 POSIX shell) |
| NVIDIA 驱动 | ≥ 570 |
| CUDA | ≥ 12.9(Blackwell/GB 系列需要 CUDA 13 世代) |
| Docker | ≥ 26,并安装 NVIDIA container toolkit |
| Docker root dir | 位于剩余空间 > 100 GB 的数据盘 |
| GPU 访问 | 容器内 `nvidia-smi` 可用 |
| 节点间 root ssh | 免密,**多机 NCCL 测试(`mpirun`)必需** |
| 共享存储 | 可选。这些脚本不依赖它 —— 但收集日志时方便 |
| 本地 NVMe / 数据盘 | 挂载路径各节点一致,建议 ≥ 3 TB |
| 网卡命名 | **各节点完全一致**(例如都是 `bond0`),RDMA 设备名同理 |
| PCIe ACS | 每个节点都关闭(`nccl_bench/close_acs.sh`),否则机内 P2P 带宽受损 |
| 持久化模式 | `nvidia-smi -pm 1` |

### 控制机

运行这些驱动脚本的机器需要 `ssh`、`sshpass`、`tar`、`python3` 和 `git-lfs`
(见下方说明)。第 2、3 关的脚本还要求**节点上**装有 `perftest`(`ib_write_bw` 等),
控制机上不需要。

> **本仓库使用 git-lfs。** `cublas_bench/GB300/gemm-memread/` 下有约 400 MB 由
> git-lfs 跟踪的 `.so` 库。未安装 `git-lfs` 时拿到的是指针文件而不是库文件。
> 请先装 git-lfs;若不需要该包,可用 `GIT_LFS_SKIP_SMUDGE=1` 克隆。

## 容器镜像

测试在 CUDA 容器内运行。使用与你的 GPU 世代匹配的镜像,或用
`nccl_bench/Dockerfile` 自行构建。

| GPU 世代 | 建议使用的镜像世代 |
| --- | --- |
| B200 / B300 / GB200 / GB300 | NVIDIA PyTorch 26.05 世代(CUDA 13) |
| H100 / H200 / H800 / H20 / A800 / L20 | NVIDIA PyTorch 25.05 世代(CUDA 12.9) |

任何已经包含用 `_mpi` 后缀构建的 `nccl-tests` 的 CUDA 镜像都可用于第 2 关。
第 1 关用的 `cublasMatmulBench_<GPU>` 二进制**随本仓库下发** —— 不在任何镜像里。

## 目录结构

```
preflight/        第 0 关 —— 节点一致性采集器与比对器
cublas_bench/     第 1 关 —— 逐卡 GEMM 基准、二进制、基线
nccl_bench/       第 2 关 —— NCCL 集合通信测试(容器 + MPI)
perftest/         第 3 关 —— RDMA 点对点与 rail 诊断
inference_bench/  可选,仅推理卡 —— 单机 Prefill/Decode 容量(草案)
tools/            ssh 打通、结果打包、汇总生成
results/          由各关卡创建;这就是你要提交的内容
VENDOR_RUNBOOK.md 完整的分步操作流程
SUBMISSION.md     验收表与提交格式
```

## 提交什么

`tools/collect_results.sh` 把四个关卡的原始日志、环境快照和清单打包成单个 tar 包,
`tools/summarize.py` 再把它变成带通过/失败判定的文字版 `SUMMARY.md`。
清单和需要填写的表格见 [SUBMISSION.md](SUBMISSION.md)。

**没有产出的关卡等于没有验收。** 打包脚本会点名缺失的关卡,而不会静默略过。

可选的推理关不在这两个脚本内:`results/inference/` 需自行打包,与 tar 包一并提交。
