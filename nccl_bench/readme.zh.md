**中文** | [English](readme.md)

# nccl_bench — NCCL 集合通信 / 互连带宽测试

使用 [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) 测试节点内 NVLink 与节点间 RDMA 带宽，覆盖对训练最重要的四类集合通信：**AllReduce、All2All、AllGather、ReduceScatter**。

这是验收流程的关卡 3。请在 [cuBLAS](../cublas_bench/readme.md)（单 GPU 计算）之后运行，避免把计算问题误判为 fabric 问题。

## 产物

每种集合通信生成一份 `nccltest_<test>.log`，以及每个测试在 16G 下的 **峰值 out-of-place busbw** 一行汇总，这是要与 [baselines.md](baselines.md) 对比的数值。

## 前置条件

- 每个节点上均已安装 Docker 与 NVIDIA container runtime。
- **节点间 root ssh，免密** —— 背景：`mpirun` 通过 ssh 拉起 rank。用 [`../tools/setup_ssh_keyless.sh`](../tools/setup_ssh_keyless.sh) 配置。
- 每个节点上 **PCIe ACS 已禁用**（`close_acs.sh`），否则节点内的 peer-to-peer DMA 被阻断，节点内带宽偏低。
- 一份节点列表，其中每个节点都能互通，且所有节点上 bootstrap 网络的接口名一致。
- 要么是已构建好 nccl-tests 的镜像（本目录的 `Dockerfile` 构建一个），要么是自带 nccl-tests 的 CUDA 镜像。

## Step 1 — 探测参数

这些脚本所需的大部分值都是集群相关的。不要猜测：

```bash
cd nccl_bench
bash nccltest_autoconf.sh -H hosts.txt      # -H also checks node reachability
bash nccltest_autoconf.sh --apply           # write the values into set_env.sh
```

它会报告每节点 GPU 数、bootstrap 接口、RDMA HCA 列表（过滤为实际与 GPU 亲和的那些）、fabric 类型、RoCE v2 GID index 以及镜像。任何无法确定的项会标记为 `[WARN]` —— 先修复这些。

> **GID index 是个真坑。** 它由驱动分配，当 IP 方案或 RoCE 版本变化时会漂移。错误的值会让每个 QP 报 `ibv_modify_qp INIT->RTR ETIMEDOUT` 失败，看起来像 fabric 故障。可随时用 `bash ../tools/detect_roce_gid.sh <dev> --table` 检查。

## Step 2 — 准备节点

```bash
cp hosts.txt.example hosts.txt && vim hosts.txt   # one address per node
bash split_ip.sh                                  # host8.txt, host16.txt, host32.txt, ...
```

`split_ip.sh` 按二次幂步长各生成一份节点列表，从而可以读取带宽随规模变化的曲线。

在每个节点上禁用 ACS（宿主机改动，之后需重启）：

```bash
for h in $(cat hosts.txt); do ssh $h 'bash -s' < close_acs.sh; done
```

## Step 3 — 在每个节点上启动容器

```bash
for h in $(cat hosts.txt); do
  ssh $h 'mkdir -p /root/nccl_bench'
  scp launch_container.sh run.sh run_nccltest.sh 1node_nccltest.sh \
      parse_nccl_log.py gpu_nic_topo.sh hosts.txt $h:/root/nccl_bench/
  ssh $h "cd /root/nccl_bench && IMAGE_NAME=<your image> HOSTS_FILE=/root/nccl_bench/hosts.txt \
          NCCL_SOCKET_IFNAME=<iface> bash launch_container.sh"
done
```

容器是 privileged 的，可见所有 GPU 与 `/dev/infiniband`，使用 host 网络，并在 7900 端口运行 sshd，以便 `mpirun` 能访问对端节点。

## Step 4 — 运行

在 **一个** 节点上执行。先跑单节点检查 —— 它把 NVLink 域与 fabric 隔离开，这样节点本地的问题不会伪装成网络问题。

```bash
# 单节点,全部四项集合通信
docker exec -i nccltest bash -c "bash /root/1node_nccltest.sh"

# 多节点,全部四项集合通信
docker exec -i nccltest bash -c \
  "HOSTS_FILE=/root/hosts_dir/hosts.txt \
   NCCL_SOCKET_IFNAME=<iface> NCCL_IB_GID_INDEX=<idx> \
   bash /root/run_nccltest.sh allreduce all2all allgather reducescatter \
   | tee /root/nccltest_all.log"
```

其他规模无需重启容器 —— 只需把容器看到的节点列表换成不同规模再重跑：

```bash
cp host16.txt hosts.txt && docker exec -i nccltest bash -c "bash /root/run_nccltest.sh all2all"
```

单独的集合通信与大小：

```bash
docker exec -i nccltest bash -c "bash /root/run_nccltest.sh allreduce"
docker exec -i nccltest bash -c "bash /root/run_nccltest.sh all2all b=8 e=4G n=20"
```

## Step 5 — 解析

```bash
docker exec -i nccltest python3 /root/parse_nccl_log.py /root/                  # peak
docker exec -i nccltest python3 /root/parse_nccl_log.py /root/ --at-size 16G    # the baseline row
docker exec -i nccltest python3 /root/parse_nccl_log.py /root/ --csv /root/nccl.csv
```

与 [baselines.md](baselines.md) 对比，同时确认每份日志都显示
`# Out of bounds values : 0` —— 非零计数意味着集合通信返回了错误结果，这比带宽更重要。

## Step 6 — 收集日志

日志位于 **容器内**。在它们落到宿主机上之前，`tools/collect_results.sh` 看不到，所以在跑测试的节点上把它们拷出来：

```bash
bash collect_logs.sh                          # -> ../results/nccl/
bash collect_logs.sh --out /tmp/nccl_logs     # or wherever you like
```

如果你在某节点上跑了测试，想把日志拉回控机：

```bash
scp -P <PORT> -r <USER>@<node>:/data/ai_infra_perf_prepare/results/nccl/* results/nccl/
```

**输出：** 多节点运行得到 `results/nccl/nccltest_<op>.log`，单节点运行得到 `1node_nccltest_<op>.log`。打包步骤期望的就是这个目录。

## Step 7 — 停止容器

```bash
for h in $(cat hosts.txt); do
  ssh -p <PORT> <USER>@$h 'docker stop nccltest >/dev/null 2>&1; docker rm nccltest >/dev/null 2>&1'
done
```

## 多节点判定规则

- **All2All 跨节点时下降到 ~10–20 %，这在每一代 GPU 上都如此。** 这是集合通信本身的工作方式，不是缺陷。把多节点 All2All 与多节点 baseline 对比，绝不要与单节点数值对比。
- **H800 AllReduce ≈215 GB/s 是硬件规格**（cut-down NVLink）。不要把它报为故障。
- 在 **rail 优化的 fabric** 上，只对比 **同一 rail/leaf** 上的节点。先用 [`../perftest/rail_isolation_check.sh`](../perftest/rail_isolation_check.sh) 验证；混用 rail 会让对比毫无意义。
- **大规模** 下降（例如到 32 节点还平，之后开始掉）指向 spine 过订阅。报告它拐弯的规模。

## 文件

| File | Purpose |
| --- | --- |
| `nccltest_autoconf.sh` | 探测所有集群相关参数；`--apply` 写入 `set_env.sh` |
| `launch_container.sh` | 在一个节点上启动 `nccltest` 容器 |
| `run.sh` | 容器入口：ssh 配置 + sshd |
| `run_nccltest.sh` | 多节点运行器（mpirun，每个 GPU 一个进程） |
| `1node_nccltest.sh` | 单节点运行器（一个进程，使用所有本地 GPU） |
| `gpu_nic_topo.sh` | GPU ↔ HCA ↔ netdev 亲和性；生成 `NCCL_IB_HCA` |
| `close_acs.sh` | 禁用 PCIe ACS（宿主机级别，需 root） |
| `split_ip.sh` | 生成二次幂节点列表 |
| `parse_nccl_log.py` | 从日志中提取峰值 / 指定大小的 busbw |
| `collect_logs.sh` | 把日志从容器拷到 `results/nccl/` |
| `baselines.md` | 参考 busbw 数值与判定规则 |
| `Dockerfile` | 构建带 nccl-tests + sshd 的镜像 |
| `set_env.example.sh` | 配置模板 |
| `hosts.txt.example` | 节点列表模板 |

## 故障排查

| Symptom | Likely cause |
| --- | --- |
| 每个 rank 打印 `LOCALIP=NONE` | `NCCL_SOCKET_IFNAME` 错误或各节点不一致 |
| `ibv_modify_qp INIT->RTR ETIMEDOUT` | `NCCL_IB_GID_INDEX` 错误，或 RoCE QoS（MTU / PFC / trust）在节点间不一致 |
| `mpirun -np` 多起一个 rank | `nvidia-smi -L` 输出多余行。`run_nccltest.sh` 通过统计 `^GPU` 行来避免此问题 |
| AllGather / ReduceScatter 仅在某些节点对上崩溃并报 `unhandled system error (all_gather.cu:50)` | `NCCL_CUMEM_HOST_ENABLE=1` 与 NVLS 冲突。脚本默认为 0 —— 不要覆盖 |
| 单节点带宽远低于 baseline | PCIe ACS 仍启用，或可见 GPU 数少于预期 |
| 多节点在某节点对上正常，在另一节点对上异常 | rail/leaf 摆放问题。用 `rail_isolation_check.sh` 检查 |
| `mpirun` 卡住或容器内 `ssh` 失败 | 容器 sshd 未运行，或节点间 7900 端口被阻断 |

## 关于 ACS 的说明

`close_acs.sh` 修改 PCIe 配置空间，需要 root 与重启才能完全生效。这是宿主机级别的改动 —— 在全集群范围内运行前，先确认它在你的环境中可被接受。
