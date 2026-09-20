**中文** | [English](VENDOR_RUNBOOK.md)

# 厂商操作手册 —— 分步指南

从上到下依次执行。每个关卡都会把结果写入 `results/<gate>/`,最后的打包脚本
会收集全部四关。**如果某一步没有产出,该关卡就没有通过验收。**

过程中请替换以下占位符:

| 占位符 | 含义 | 示例 |
| --- | --- | --- |
| `<IMAGE>` | CUDA 容器镜像(见[选择镜像](#选择镜像)) | `nvcr.io/nvidia/pytorch:25.05-py3` |
| `<GPU>` | GPU 型号,需与基线表对应 | `H800`、`H20`、`B200` |
| `<NODE_DIR>` | 本仓库在**节点上**的路径 | `/data/ai_infra_perf_prepare` |
| `<PORT>` | 节点的 SSH 端口 | `22` |
| `<USER>` | 节点上的 SSH 用户 | `root` |

除特别说明外,所有命令都在**控制机**上运行 —— 即那台能 ssh 到所有节点的机器。

---

## 第 0 步 —— 准备控制机

```bash
sudo bash install_env.sh
```

安装 `ssh`、`sshpass`、`rsync`、`python3`、`tar` 和 **`git-lfs`**。git-lfs 这一项
很重要:`cublas_bench/GB300/gemm-memread/` 是用它跟踪的,没有它你拿到的会是指针
文件而不是库文件。

## 第 1 步 —— 编写节点清单

```bash
cp preflight/hosts.txt.example preflight/hosts.txt
vim preflight/hosts.txt
```

每行一个条目,允许 `#` 注释。每个条目就是你 ssh 该节点时用的东西 —— 可以是裸地址
(`10.0.0.11`),也可以是 `~/.ssh/config` 里的别名(`node01`);用别名时,端口、用户
和跳板机都来自那份配置,后续步骤里的 `-P` / `-u` 参数就可以省掉。节点不能直连时,
别名是更省事的选择。

这份清单供第 0、1、3 关以及第 6 步的控制循环使用。**第 2 关需要另一份不同的地址
清单** —— 见 6b。

## 第 2 步 —— 打通免密 ssh

```bash
NODE_PASSWORD='<node password>' \
  bash tools/setup_ssh_keyless.sh preflight/hosts.txt --port <PORT> --user <USER>
```

每一行都应打印 `[ OK ]`。重复执行是安全的 —— 公钥是追加 + 去重,绝不覆盖。

## 第 3 步 —— 把仓库分发到每个节点

第 1、2 关需要节点上有本仓库。第 0 关和第 3 关不需要(它们通过 ssh 管道下发脚本),
但在这里统一做一次能让后面更省心。

```bash
NODES=$(grep -vE '^[[:space:]]*(#|$)' preflight/hosts.txt | awk '{print $1}')

for h in $NODES; do
  ssh -p <PORT> -o BatchMode=yes <USER>@$h "mkdir -p $(dirname <NODE_DIR>)"
  rsync -a --delete \
        --exclude '.git' --exclude 'cublas_bench/GB300' --exclude 'results' \
        --exclude '__pycache__' --exclude '*.log' --exclude '*.tar.gz' \
        ./ <USER>@$h:<NODE_DIR>/
  echo "$h done"
done
```

检查一下:

```bash
ssh -p <PORT> <USER>@<first node> "ls <NODE_DIR>"
```

记牢你实际用的路径 —— 第 1、2 关都要把它作为参数传入。

---

## 第 4 步 —— 第 0 关:节点一致性

**为什么先做:** `driver_version` 一致**并不代表**节点同构。这里的配置漂移会静默地
让后续所有对比失效。

```bash
mkdir -p results/preflight

bash preflight/cluster_runtime_check.sh \
     -H preflight/hosts.txt -P <PORT> -u <USER> \
     -o results/preflight/runtime.csv

bash preflight/cluster_spec_check.sh \
     -H preflight/hosts.txt -P <PORT> -u <USER> \
     -o results/preflight/spec.csv
```

**预期:** 两者都退出 `0`,且 `results/preflight/` 下有两个 CSV。

**若某个关键项(`★`)存在差异 → 退出码 2。** 把该节点从 `preflight/hosts.txt` 中
排除,重跑两者,并记录排除了哪些节点、为什么。不要静默丢弃节点。

> 计算 HCA 集合是按"所有 ACTIVE 端口的 HCA"自动探测的,可能把存储/管理网卡也包含
> 进来。对一致性检查而言这没问题。若想把范围限定在训练 fabric 上,设置
> `COMPUTE_HCA_RE` 和/或 `EXCLUDE_HCA` —— 见 `preflight/readme.md`。

---

## 第 5 步 —— 第 1 关:单卡算力(cuBLAS)

### 5a. 先做快速验证(单节点,只测 0 号卡)

在投入全集群扫描之前先做这个 —— 它能验证镜像和二进制。用测试将要运行的那个节点,
或任意一个节点:

```bash
cd cublas_bench
GPU_NAME=<GPU> image=<IMAGE> bash auto_cublas_bench_by_docker.sh
cd ..
```

**预期:** 输出 6 个精度的表格,以 `Result: ALL PASS` 结尾,退出码 `0`。

> 这只压测了 **0 号卡**。"这个节点通过了"并不意味着 8 张卡都通过了 —— 那是 5b 的
> 职责。

### 5b. 全集群,逐卡

```bash
bash cublas_bench/cluster_cublas_percard.sh \
     -H preflight/hosts.txt -P <PORT> -u <USER> \
     -i <IMAGE> -g <GPU> \
     -s <NODE_DIR>/cublas_bench
```

`-s` 是 **节点上** `cublas_bench` 的路径 —— 即第 3 步用的那个路径。写错的话运行会
立刻以 `no_bench_script` 失败。

耗时为每卡约 5 分钟,8 卡节点约 40 分钟。若只想检查连通性和镜像而不实际跑负载,
先加 `--preflight`。

**预期:** `results/cublas/percard_cluster/<node>/dev<N>.log`(每卡一个文件),
外加 `fleet_percard.csv` 和离群报告。

**怎么判读:** 核心看**全集群一致性** —— 一张卡在多个精度上一致偏低,才是弱卡。
跨机极差 ~5–6 % 属正常;单机内部应低于 ~3 %。单个精度略低于基线,通常是 shape
导致的 artifact,而不是坏板子。

> **忽略日志里的 `libcompat_fix.so ... cannot be preloaded`** —— 那是宿主机的
> `/etc/ld.so.preload` 漏进了容器。
>
> **H800 通过 cuBLAS `sss_fast_tf32` 测 TF32,部分板子会读到约半速。这是 cuBLAS
> 的 artifact,不是弱卡** —— 真实框架走的是 `allow_tf32`,同一块板子是满速的。
> H800 的 TF32 要看 torch 的 `allow_tf32`,绝不要用 `sss_fast_tf32` 的数字判定。

---

## 第 6 步 —— 第 2 关:NCCL 互联

### 6a. 推导集群相关参数

这一步要**在某个节点上**运行,而不是在控制机上 —— 它要读取该节点的 GPU 和 RDMA
设备信息。

```bash
FIRST=$(grep -vE '^[[:space:]]*(#|$)' preflight/hosts.txt | awk '{print $1}' | head -1)

ssh -p <PORT> <USER>@$FIRST \
  'cd <NODE_DIR>/nccl_bench && bash nccltest_autoconf.sh --apply'
```

读取输出,把标了 `[WARN]` 的问题先解决掉再继续。**记下 `NETWORK_IF`、
`NCCL_IB_HCA` 和 `NCCL_IB_GID_INDEX`** —— 6c–6e 都要用。`--apply` 会在节点上写入
`set_env.sh`。

### 6b. 编写 **bootstrap** 主机清单

这里存在**两个不同的地址空间**,把它们搞混是这一关最常见的出错方式:

| 清单 | 内容 | 使用者 |
| --- | --- | --- |
| `preflight/hosts.txt`(第 1 步) | **你** ssh 用的地址 | 第 0、1、3 关以及下面的控制循环 |
| `nccl_bench/hosts.txt`(本步) | 各节点**在 `NETWORK_IF` 上**的地址 | 容器内的 `mpirun` 和 NCCL |

若你的 ssh 通路与训练网络是同一条,两份清单完全相同,直接复制文件即可。若不同
—— 例如你经跳板机访问节点 —— 那它们**并不相同**:

```bash
cp hosts.txt.example hosts.txt
vim hosts.txt          # 每行一个 NETWORK_IF 地址,顺序保持一致
```

> 处于跳板机之后,只影响 **ssh** 那份清单。因此 6c–6e 中的命令统一用
> `preflight/hosts.txt` 来 ssh(别名在那里可用,端口和跳板机由 ssh 配置提供),
> 而容器被告知使用 `nccl_bench/hosts.txt`。**不要**去 ssh bootstrap 地址。

### 6c. 在每个节点启动容器

```bash
# 这里的 ssh 用 SSH 清单;容器的 HOSTS_FILE 用 BOOTSTRAP 清单
SSH_NODES=$(grep -vE '^[[:space:]]*(#|$)' preflight/hosts.txt | awk '{print $1}')
IMAGE_NAME=<IMAGE>
NCCL_SOCKET_IFNAME=<NETWORK_IF from 6a>

for h in $SSH_NODES; do
  echo "=== $h ==="
  ssh <SSHOPTS> $h "mkdir -p <NODE_DIR>/nccl_bench"
  rsync -a -e '<SSHCMD>' nccl_bench/hosts.txt $h:<NODE_DIR>/nccl_bench/hosts.txt
  ssh <SSHOPTS> $h \
    "cd <NODE_DIR>/nccl_bench && \
     IMAGE_NAME=$IMAGE_NAME HOSTS_FILE=<NODE_DIR>/nccl_bench/hosts.txt \
     NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME bash launch_container.sh >/dev/null 2>&1 && \
     docker ps --format '{{.Names}}' | grep -q nccltest && echo 'container up'"
done
```

`<SSHOPTS>` 按你的环境填写,例如 `-p 22 -o BatchMode=yes`;若 ssh 配置里已经带了
端口和跳板机,则什么都不用写。`-e '<SSHCMD>'` 是 `rsync` 的对应写法,例如
`-e "ssh -p 22"`。

**预期:** 每个节点都输出 `container up`。

### 6d. 单机 NVLink 测试

在任意一个节点上(这里取 SSH 清单里的第一个):

```bash
FIRST=$(grep -vE '^[[:space:]]*(#|$)' preflight/hosts.txt | awk '{print $1}' | head -1)

ssh <SSHOPTS> $FIRST \
  'docker exec -i nccltest bash -c "bash /root/hosts_dir/1node_nccltest.sh"'
```

先跑这个再跑多机测试:它把 NVLink 域与 fabric 隔离开,这样机内问题不会伪装成
网络问题。

**预期:** 四项集合通信各自以 `[<op>] exit=0` 和 `# Out of bounds values : 0` 结尾。

### 6e. 多机测试

```bash
ssh <SSHOPTS> $FIRST \
  'docker exec -i nccltest bash -c \
     "HOSTS_FILE=/root/hosts_dir/hosts.txt \
      NCCL_SOCKET_IFNAME=<NETWORK_IF> NCCL_IB_GID_INDEX=<GID index from 6a> \
      bash /root/run_nccltest.sh allreduce all2all allgather reducescatter"'
```

**预期:** 四项都报告成功,且各自 `# Out of bounds values : 0`。

要测其他集群规模,在**节点上**编辑 `nccl_bench/hosts.txt`(它是 bind mount 的,
容器会立即读到改动)然后重跑 6e —— 无需重启容器:

```bash
ssh <SSHOPTS> $FIRST "vim <NODE_DIR>/nccl_bench/hosts.txt"
```

### 6f. 把日志取出来

仍在 `$FIRST` 上 —— 日志在容器里,不落到宿主机上,打包脚本是用不到的:

```bash
ssh <SSHOPTS> $FIRST \
  'cd <NODE_DIR>/nccl_bench && bash collect_logs.sh'

# 再拉回控制机
rsync -a -e '<SSHCMD>' $FIRST:<NODE_DIR>/results/nccl/ results/nccl/
```

**预期:** `results/nccl/` 中,多机测试对应 `nccltest_<op>.log`,单机测试对应
`1node_nccltest_<op>.log`。

### 6g. 停止容器

```bash
for h in $SSH_NODES; do
  ssh <SSHOPTS> $h 'docker stop nccltest >/dev/null 2>&1; docker rm nccltest >/dev/null 2>&1'
done
```

**怎么判读:** 拿 16G 的峰值,与 `nccl_bench/baselines.md` 中相同 GPU / 单机卡数 /
机器数的行对比。

- **跨机 All2All 会跌到单机数值的 ~10–20 %**,这在所有 GPU 世代上都是如此。
  这是该集合通信本身的工作方式 —— 绝不要拿多机 All2All 去比单机数字。
- **H800 AllReduce ≈215 GB/s 是 NVLink 被削减后的硬件规格**,不是故障。
- 在 rail-optimized fabric 上,只能对比同一 rail/leaf 内的节点 —— 第 7 步会告诉你
  这个约束是否适用。

---

## 第 7 步 —— 第 3 关:RDMA 点对点

选两个节点。`perftest`(`ib_write_bw` 等)必须装在**节点上**,且那里的
`ulimit -l` 必须是 `unlimited`。

### 7a. RDMA 是否被限制在单条 rail 内?

```bash
mkdir -p results/perftest
bash perftest/rail_isolation_check.sh <nodeA> <nodeB> --ssh-port <PORT> --ssh-user <USER> \
  | tee results/perftest/rail_isolation.txt
```

两个节点按你 ssh 它们的方式填写(地址或别名)。**不要传设备参数** —— 脚本会自动
挑选与 GPU 亲和的计算 rail。若你的节点对 rail 的命名方式不同,请显式传入
`devA devB`,否则会得到误导性的结论。

**预期:** 一行结论。若判定为 **rail 隔离**,那么从这一步起,只能在同一个
rail/leaf 内对比节点,并且要知道跨机流水线并行对 rail 配对很敏感。

### 7b. 逐 rail 明细

```bash
bash perftest/perftest_pair.sh <nodeA> <nodeB> --ssh-port <PORT> --ssh-user <USER>
```

**预期:** 一张逐 rail 的表。200G 端口单 QP 应读到 **~193 Gb/s**,双向约为 2 倍,
时延 2.5–2.9 µs。**五个错误计数器必须全为 0** —— 在原始日志里搜
`hw_counters snapshot`。

> bond 聚合口上的单个 QP 会被 LAG 哈希钉在其中一个物理口上,所以 2×200G 的 rail
> 读到 ~193 Gb/s 是正确的,而不是"半条坏掉的 rail"。真正的容量由 7c 测量。

### 7c. 逐 rail 聚合

```bash
bash perftest/perftest_agg.sh <nodeA> <nodeB> --ssh-port <PORT> --ssh-user <USER>
```

**预期:** 2×200G 的 bond rail 上应达到 **~390 Gb/s peak**,而不是 ~195。判定用
≥3 次中的 **peak** —— 某一次读到一半只是 LAG 哈希运气,不是故障。

**预期产物:** 三个脚本都会写入 `results/perftest/`
(`summary_<ts>.txt`、`summary_agg_<ts>.txt` 以及原始日志)。

---

## 第 8 步 —— 打包并提交

```bash
bash tools/collect_results.sh --dir results --vendor <vendor name> --gpu <GPU>
python3 tools/summarize.py results --vendor <vendor name> --gpu <GPU>
```

`collect_results.sh` 会检查四个关卡的目录。**只要有一个是空的,它就会拒绝打包并
点名该关卡** —— 那一关没有通过验收。随后它会写出 `results/environment.txt`、
`results/MANIFEST.txt` 和 tar 包。

`summarize.py` 会写出 `results/SUMMARY.md` 和 `results/results.json`,包含逐关卡
判定和总体判定。

**预期的最终状态:**

```
results/
  SUMMARY.md          已生成
  results.json        已生成
  MANIFEST.txt        已生成
  environment.txt     已生成
  preflight/          runtime.csv, spec.csv
  cublas/             percard_cluster/<node>/dev<N>.log, fleet_percard.csv
  nccl/               nccltest_<op>.log, 1node_nccltest_<op>.log
  perftest/           summary_<ts>.txt, summary_agg_<ts>.txt, perftest_<ts>.log
```

提交 tar 包,以及 **[SUBMISSION.md](SUBMISSION.md)** 中填写完成的表格。

---

## 选择镜像

你需要一个 CUDA 容器镜像。可以用 `nccl_bench/Dockerfile` 自行构建(它会加入
nccl-tests 和 sshd),或直接使用对应世代的 NVIDIA PyTorch 镜像:

| GPU 世代 | 镜像世代 |
| --- | --- |
| B200 / B300 / GB200 / GB300 | CUDA 13 世代(`nvcr.io/nvidia/pytorch:26.05-py3` 或类似) |
| H100 / H200 / H800 / H20 / A800 / L20 | CUDA 12 世代(`nvcr.io/nvidia/pytorch:25.05-py3` 或类似) |

第 2 关要求镜像里已经包含用 `_mpi` 后缀构建的 nccl-tests(本仓库的 `Dockerfile`
已经做了这件事)。第 1 关的 `cublasMatmulBench_<GPU>` 二进制**随本仓库下发**,
任何镜像里都没有。

---

## 容易踩的坑

| 现象 | 原因 |
| --- | --- |
| 第 2 关:每个 rank 都打印 `LOCALIP=NONE` | `NCCL_SOCKET_IFNAME` 填错,或各节点上该网卡名不一致 |
| 第 2 关:`ibv_modify_qp INIT->RTR ETIMEDOUT` | `NCCL_IB_GID_INDEX` 填错,或节点间 RoCE QoS 设置(MTU / PFC / trust)不一致 |
| 第 2 关:AllGather/ReduceScatter 只在部分节点对上崩 `all_gather.cu:50` | `NCCL_CUMEM_HOST_ENABLE=1` 与 NVLS 冲突。脚本已默认设为 0 —— 不要覆盖 |
| 第 1 关:`no_bench_script` | `-s` 指向的节点路径不对 —— 见第 3 步 |
| 第 3 关:每条 rail 都被跳过,提示 "no rail IP" | 不应发生;请带上 `ibdev2netdev` 的输出上报 |
| 第 3 关:`ibv_reg_mr` failed | 该节点的 `memlock` 不是 unlimited |
| `collect_results.sh` 报某关卡缺失 | 该关卡没有在预期位置产出文件 —— 重读它对应的步骤 |

---

## 各部分的验证情况

第 0–8 步已在 2026-09-20 于一个双节点 H20 集群(每节点 8× H20,RoCE,
GID index 3)上**按本文档原样执行**。四个关卡都自动在 `results/<gate>/` 下产出了
对应产物,无需手工搬移文件;`collect_results.sh` + `summarize.py` 端到端跑通。

未在该集群上验证、建议你在自己的环境首跑时留意的部分:

- `tools/setup_ssh_keyless.sh` —— 该环境本来就已配好免密。
- `nccl_bench/close_acs.sh` —— 它会修改 PCIe 配置且需要重启,因此刻意未执行。
- 第 1 关在单节点上以**完整 8 卡规模**跑过,结果干净(全集群极差 0.06–1.7 %,
  无离群卡)。对本文档命令的复验为节省时间用了每节点 1 卡 —— 机制完全相同。
- 第 2 关跑过 1 节点(8 GPU)和 2 节点(16 GPU);未测更大规模。
