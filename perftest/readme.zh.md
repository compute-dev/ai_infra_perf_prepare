**中文** | [English](readme.md)

# perftest — RDMA 点对点 baseline

独立于 NCCL 测量两节点之间的原始 RDMA fabric。这是把 **fabric** 问题与 **NCCL / 配置** 问题分开的方法：如果点对点跑满线速，则 fabric 正常，故障在它之上。

与 [nccl_bench](../nccl_bench/readme.md) 互补，后者测量 NCCL 构建于同一 fabric 之上时集合通信实际达到的水平。

## 四个脚本

| Script | 它回答的问题 |
| --- | --- |
| `perftest_pair.sh` | 按 rail：write / read 带宽、双向带宽、write 与 send 延迟，以及 RDMA 错误计数器。找出死端口或降速端口。 |
| `perftest_agg.sh` | 按 rail：用单进程中的多个 QP 测 **aggregate** 容量。这才是 NCCL 实际使用的。 |
| `rail_isolation_check.sh` | RDMA 是否被限制在单条 rail 内？决定节点能否自由对比与分组。 |
| `diag_gid.sh` | RoCE GID index 是连接失败的原因吗？ |

## 前置条件

- **perftest**（`ib_write_bw`、`ib_read_bw`、`ib_write_lat`、`ib_send_bw`、`ib_send_lat`）在 **两** 个节点上均已安装。
- **memlock unlimited** 在两节点上均生效（`ulimit -l`），否则 `ibv_reg_mr` 失败，所有测试报错。用 `ulimit -l` 检查；在 `/etc/security/limits.conf` 设置（`* hard memlock unlimited`）并重新登录。
- 从你运行这些脚本的机器能 ssh 到两个节点。

如果主机上未安装 perftest，在带它的容器内运行，启动参数为 `--network=host --privileged --device=/dev/infiniband
--ulimit memlock=-1`。

## 快速开始

```bash
# 1. RDMA 是否被限制在单条 rail 内?先做这个 —— 它决定下面一切怎么分组。
bash rail_isolation_check.sh node01 node02

# 2. 逐 rail 点对点明细,双向,带错误计数器快照
bash perftest_pair.sh node01 node02

# 3. 逐 rail 聚合容量(多 QP,取 3 次中的 peak 判定)
bash perftest_agg.sh node01 node02
```

设备会自动发现（每个有 ACTIVE 端口的设备）。用 `--devices "mlx5_0 mlx5_1"` 覆盖；用 `--ssh-port 2222` 设置 ssh 端口；用 `--gid 3` 设置 RoCE GID index（否则自动探测）。

## 如何解读结果

### 单 QP（`perftest_pair.sh`）

与 **一个端口的线速** 对比，而非整条 rail：

- 200G 端口应读 **~193 Gb/s**（~24.2 GB/s，≈96.7 % 线速）。双向应一致。
- 400G 端口应读 ~389 Gb/s。
- 双向 write 约为单向数值的 **2×**。
- **write 与 send 延迟：2.5–2.9 µs**（健康的 200G rail）。

**在 bond 聚合口 / 802.3ad / LAG 的 rail 上，单 QP 被 LAG 哈希固定到一条物理端口。** 所以在 2×200G bond 上单 QP 读数 ~193 Gb/s 是正确且符合预期的 —— 它 *不是* 一条坏 rail 的一半。用 `perftest_agg.sh` 获取 rail 的真实容量。

### Aggregate（`perftest_agg.sh`）

- 2×200G bond rail 应达到 **~390 Gb/s 峰值**（不是 ~195）。
- **以峰值判定，且至少跑 3 次取最大值。** 当所有 QP 都哈希到同一条成员端口时，单次运行合法地只报约一半。这是哈希运气。如果一条 rail 在所有运行中都停留在一半，那才是真正的问题 —— 检查该成员端口的链路状态与线缆。
- 在 bond rail 上 `avg` 常常远低于 `peak`，因为哈希把 QP 不均匀地分散开。对比峰值。

> **不要** 通过把多条独立短单 QP 流的峰值相加来推导 "aggregate capacity"。那样会高报 —— 可能超过 rail 的物理上限，因为短运行永远达不到协调的稳态。`perftest_agg.sh` 使用正确的方法。

### 错误计数器 —— 每次都要检查

`perftest_pair.sh` 在运行前与每个方向之后，为每条 rail 在两个节点上快照五个计数器：

```
out_of_sequence   packet_seq_err   req_cqe_error
local_ack_timeout_err   rnr_nak_retry_err
```

**所有五个在所有位置都必须为 0。** 任何非零值都指明是哪个节点、哪条 rail 有故障。一条 rail 可能在带宽上看起来很快，却仍在丢包或乱序 —— 这会在后期表现为训练不稳定，而非此刻的低带宽。在原始日志中 grep `hw_counters snapshot`。

### Rail 隔离

| 同 rail | 跨 rail | 含义 | 怎么办 |
| --- | --- | --- | --- |
| 满线速 | 连不上 | **rail 隔离** | 只在同一 rail/leaf 内对比带宽；跨节点流水线并行对配对敏感；从网络侧获取一份 "HCA 端口 → leaf" 映射 |
| 满线速 | 可用 | 非 rail 隔离 | 自由分组；按 leaf 过订阅判定 |
| 失败 | — | 同 rail baseline 未跑 | 不是 rail 隔离结果 —— 检查设备名、GID 与布线 |

**不要用 ping 来判定。** IP 层跨 rail 可达，会误导你；被阻断的是 RoCE 路径。

> `rail_isolation_check.sh` 刻意替换 **对端的 `-d` 设备**，而不仅是 IP 地址。RDMA 走哪条 rail 由两端的 `-d` 决定；仅改 IP 会静默地停留在同一条 rail 上，产生假的 "跨 rail 可用"。该脚本还会验证对端 server 是否真的启动了，因为一个拼错的设备名看起来与 "rail 隔离" 完全一样。

## 输出

三个脚本默认都写入 `results/perftest/`，即 `tools/collect_results.sh` 打包的目录：

| File | 来自 |
| --- | --- |
| `perftest_<ts>.log` | `perftest_pair.sh` —— 原始输出加 `hw_counters snapshot` 块 |
| `summary_<ts>.txt` | `perftest_pair.sh` —— 按 rail 的表格，每个方向一块 |
| `perftest_agg_<ts>.log` | `perftest_agg.sh` —— 每次运行的原始输出 |
| `summary_agg_<ts>.txt` | `perftest_agg.sh` —— 按 rail 各次运行的 peak/avg |

用 `--results DIR` 覆盖。`rail_isolation_check.sh` 仅输出到 stdout —— 想放进打包结果就自己保存：

```bash
bash rail_isolation_check.sh node01 node02 --ssh-port 22 \
  | tee results/perftest/rail_isolation.txt
```

## 故障排查

| Symptom | Likely cause |
| --- | --- |
| `ibv_reg_mr` 失败 | 该节点 memlock 未 unlimited |
| `ibv_modify_qp INIT->RTR ETIMEDOUT`（IB 与 RoCE 都出现） | GID index 错误 —— 跑 `diag_gid.sh` |
| `Couldn't connect` / 没有监听 | Server 未启动：设备名错误，或端口上有残留进程。脚本用 `pkill -x`（精确名）；`pkill -f` 会把 ssh 会话本身杀掉 |
| bond 上单 QP ≈ rail 标称速率的一半 | LAG 预期行为 —— 见上文 |
| 跨 rail 可用但慢 | rail 未隔离；多半是 leaf 过订阅 |
| 带宽正常但计数器增长 | 该 rail 的物理层问题 —— 线缆、光模块或端口 |

## 文件

| File | Purpose |
| --- | --- |
| `perftest_pair.sh` | 按 rail，双向，带错误计数器快照 |
| `perftest_agg.sh` | 按 rail 的 aggregate 容量，多 QP，N 次运行的峰值 |
| `rail_isolation_check.sh` | 同 rail vs 跨 rail 判定 |
| `diag_gid.sh` | RoCE GID-index 诊断（在两节点上 dump 表） |
