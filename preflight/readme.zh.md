**中文** | [English](readme.md)

# 节点一致性预检

本目录包含一组 Bash 脚本，用于在任何性能测试（cuBLAS / NCCL / bitwise）运行
*之前* 校验 GPU 集群中每个节点是否**配置一致**。先运行它们；它们开销很小，并且能
捕获那种否则会使后续所有数据失效的隐性漂移。

---

## 为什么需要预检，以及为什么它要最先运行

两台节点上 `nvidia-smi` 显示相同的 `driver_version` 并**不**意味着这两台节点
匹配。驱动版本只是一行信息；真正决定一次 cuBLAS 或 NCCL 运行是否具有可比性的，
是驱动版本从未暴露的数十项运行时状态字段：

- 实际解析到的 userspace `libcuda.so`（cuda-compat 可能抢占它），
- ECC 模式及累计不可纠正错误，
- GPU 时钟上限与功耗上限（降频卡/弱卡会使所有 cuBLAS 基线失真），
- 计算 HCA 的在位 / ACTIVE / 速率 / 固件，
- OFED、FabricManager、peermem，
- 内核 / OS / persistence / memlock / NUMA，
- RoCE 无损配置（trust + PFC）、GPU-NIC rail 拓扑、NVLink。

其中任何一项在单台节点上发生漂移，都会静默地使对比失效。预检在每个节点上收集
所有这些字段，将每台主机上的多卡取值折叠为一个跨主机可比的标量，然后取
**逐字段的众数**作为基线并标记离群节点。在第一次 kernel launch 之前运行它；
修复或排除离群节点；之后才运行 bench。

---

## 文件

| file | role |
|------|------|
| `collect_runtime_state.sh`   | 单节点软件/运行时状态采集器（B1-B8） |
| `collect_server_spec.sh`     | 单节点硬件规格采集器（POC 验收） |
| `cluster_runtime_check.sh`   | 针对运行时状态字段的集群一致性检查 |
| `cluster_spec_check.sh`      | 针对硬件规格字段的集群一致性检查 |
| `compute_gpu_pciesw.sh`      | 每个 GPU 的 PCIe-switch 拓扑（Gen / width / vendor） |
| `hosts.txt.example`          | 带注释的主机列表模板 |

所有脚本都已 `chmod +x`。

---

## 一行命令入口

```bash
# 1. 单节点(先在任意一个节点上跑一遍,验证采集器是否正常):
bash collect_runtime_state.sh          # human-readable table
bash collect_server_spec.sh            # human-readable spec table
bash collect_runtime_state.sh -j       # single-line JSON (what the cluster checker parses)
bash collect_server_spec.sh -j

# 2. 全集群(在一台能 ssh 到清单中所有节点的主机上运行):
bash cluster_runtime_check.sh -H hosts.txt                      # default: root@...:22
bash cluster_runtime_check.sh -H hosts.txt -P 2222 -u admin      # custom port / user
bash cluster_runtime_check.sh -H hosts.txt -o rt.csv             # also write CSV

bash cluster_spec_check.sh -H hosts.txt                         # default: root@...:22, 64-way
bash cluster_spec_check.sh -H hosts.txt -p 128 -P 2222 -u admin # -p = parallelism, -P = ssh port
bash cluster_spec_check.sh -H hosts.txt -o spec.csv -k          # CSV + keep per-node raw JSON

# 3. 单节点的 PCIe switch 拓扑:
bash compute_gpu_pciesw.sh            # summary table
bash compute_gpu_pciesw.sh -v         # verbose topology chain
```

传输方式**仅使用 plain ssh**。采集器通过 stdin 管道传给每个节点的 bash
（`ssh ... 'bash -s -- -j'`），因此不需要共享文件系统、不需要 parallel-ssh
fan-out 工具，也不需要 conda。`hosts.txt` 每行一个可 SSH 到达的地址；`#` 注释行
和空行会被忽略。

### 输出

这两份 CSV **就是** 关卡 0 的交付物——它们是集群同质性的证据。把它们写入 results
目录树，以便打包步骤能找到它们：

```bash
mkdir -p results/preflight
bash cluster_runtime_check.sh -H hosts.txt -P 22 -u root -o results/preflight/runtime.csv
bash cluster_spec_check.sh    -H hosts.txt -P 22 -u root -o results/preflight/spec.csv
```

当所有关键项都一致时 `cluster_runtime_check.sh` 以 `0` 退出；当某个关键项不一致
或某节点不可达时以 `2` 退出。传给 `-H` 和 `-o` 的路径相对于你调用脚本的位置。

---

## 两个采集器各自采集什么

### `collect_runtime_state.sh` —— B1-B8 层

| layer | what | why it matters |
|-------|------|-----------------|
| B1 | driver / CUDA-API / userspace libcuda / cuda-compat | bitwise + perf 关键项 |
| B2 | ECC 模式 + 累计不可纠正错误 | 边缘 / 退化卡 |
| B3 | GPU 时钟上限 / 功耗上限 / 降频 | cuBLAS 弱卡基线 |
| B4 | 每个计算 HCA 的在位 + ACTIVE + 速率 + 固件 | 多机链路建立 |
| B5 | OFED / fabricmanager / fabric 状态 / peermem | 多机 & RDMA |
| B6 | 内核 / OS / persistence / memlock / NUMA | 调试 + bitwise 记录 |
| B7 | 容器 bench image id | TFLOPs 可比性 |
| B8 | Fabric 类型 / RoCE QoS / GPU-NIC 拓扑 / NVLink | NCCL 前置必检 |

一台主机上的多卡 / 多 HCA 取值会被折叠为一个标量：全部相同 -> 取该值；为空 ->
`N/A`；存在差异 -> `mixed:v1|v2|..`。

### `collect_server_spec.sh` —— 规格层

GPU（型号 x 数量），CPU（型号 + 核数 x socket），Memory（类型 / 每 DIMM / 每
socket / 总量），Storage-OS（系统盘容量 x 数量），Storage-Data（数据盘容量 x
数量），NIC-compute（ConnectX 型号 + 实测链路速率 x 端口数 + 芯片族规格上限），
NIC-storage（BlueField/DPU x 数量）。

> **JSON key 不对称（有意为之，不要去"修"）：** `collect_server_spec.sh
> -j` 输出 `gpu_count`、`cpu_sockets` 和 `cores_per_socket` 为**裸数字**，
> 而字符串字段是带引号的。规格比较器只读取带引号的字符串字段。这是设计如此，
> 保持原样。

---

## 比较器如何工作

两个 `cluster_*_check.sh` 脚本共用同一套算法：

1. 通过 ssh 把采集器管道到每个节点，每个节点收集一行单行 JSON。
2. 对每个字段，取**众数**（跨可达节点出现频率最高的值）作为基线。
3. 标记每个取值与基线不同的节点。

### 运行时检查器 —— 关键项 vs 参考项（`cluster_runtime_check.sh`）

标记为**关键项**的字段（在报告中以 `*` 显示）只要出现任何差异就会导致
**exit 2**：

`driver, cuda_api, libcuda, cuda_compat, gpu_count, ecc_mode, sm_clock_max,
power_limit, fabric_type, compute_hca, compute_active, compute_rate,
compute_mtu, compute_pcie, compute_fw, roce_gid_v2idx, roce_trust,
roce_pfc_prio, ofed, fabricmanager, fm_active, peermem, kernel,
gpu_nic_rail`

参考项字段只告警（不 FAIL）：

`gpu_nic_topo, lldp_switch, nvlink_links, nvlink_speed, hca_devices, os,
persistence, memlock, numa_nodes, ecc_uncorr, bench_image_id`

（`ecc_uncorr` 是参考项，但如果每个节点都在某个非零值上一致，它仍会就可能的
ECC 退化发出告警。）

退出码：`0` = 所有关键项一致；`2` = 某个关键项不一致或某节点不可达；`1` = 参数 /
依赖错误。

### 规格检查器（`cluster_spec_check.sh`）

比较七个带引号的字符串字段（`gpu, cpu, memory, storage_os,
storage_data, nic_compute, nic_storage`）。任何差异或任何不可达节点 ->
exit `2`；全部一致 -> exit `0`。

---

## 旋钮：`COMPUTE_HCA_RE` / `EXCLUDE_HCA` / `BENCH_IMAGE_RE`

这些是 `collect_runtime_state.sh` 读取的环境变量。可以在命令行上设置，也可以
为集群检查器 export。

### `COMPUTE_HCA_RE` —— 限定计算 fabric

当**已设置**时，它被用作一个**锚定 ERE**，按名称选择计算 HCA，例如：

```bash
COMPUTE_HCA_RE='mlx5_10[1-8]' bash cluster_runtime_check.sh -H hosts.txt
```

当**未设置**（默认）时，脚本**自动检测**：它取 `/sys/class/infiniband/` 下所有
`ports/1/state` 为 `ACTIVE` 的设备，再从中移除 `EXCLUDE_HCA` 列出的项。

> **存储网卡陷阱。** 自动检测出的集合可能把存储 / 管理 NIC 与训练 fabric 一并
> 包含进来。这对于一致性检查而言是*可接受的*——离群节点仍会被捕获——但它比较的
> 是错误的人群。当你只想把检查限定在训练 fabric 上时，请**显式设置
> `COMPUTE_HCA_RE`**。

### `EXCLUDE_HCA` —— 从自动检测集合中剔除名称

逗号分隔的**精确** HCA 名称，约定与
`nccl_bench/gpu_nic_topo.sh` 相同：

```bash
EXCLUDE_HCA=mlx5_1,mlx5_6 bash cluster_runtime_check.sh -H hosts.txt
```

仅在 `COMPUTE_HCA_RE` 未设置时才参考它。

### `BENCH_IMAGE_RE` —— 限定 bench image

当**已设置**时，`collect_runtime_state.sh` 用该模式去 grep `docker images` 以
解析 `bench_image_id`。当**未设置**时，该字段被标记为 `not-scoped`，而不是去
匹配任何产品或 registry 名称——没有任何 image 名称被硬编码：

```bash
BENCH_IMAGE_RE='mybench' bash cluster_runtime_check.sh -H hosts.txt
```

---

## 如何阅读这两份 CSV

`-o rt.csv`（运行时）和 `-o spec.csv`（规格）每个节点各占一行：

```
host,reachable,<field1>,<field2>,...
```

- `host` —— 来自 `hosts.txt` 的地址。
- `reachable` —— 若已采集到 JSON 则为 `1`，若节点不可达或采集器失败则为 `0`。
- 每个字段一列，存放采集器输出的精确标量（折叠后的值，或 `N/A`，或运行时 CSV
  的 `mixed:...`）。

用 shell 工具快速分诊：

```bash
# 哪些节点在某个关键项上偏离了众数?
awk -F, 'NR==1{for(i=1;i<=NF;i++)h[i]=$i} NR>1{print $1,$(diff_col)}' rt.csv

# 可达节点数
awk -F, 'NR>1{c[$2]++} END{for(k in c)print k,c[k]}' rt.csv
```

---

## 不可达节点会被计数，绝不会被静默丢弃

如果某节点没有产出 JSON（ssh 失败、采集器崩溃），检查器会将其标记为
`reachable=0`，在报告中的 "unreachable nodes
(N/M)" 下列出它，并强制 exit `2`。**不要为了让检查通过就把不可达节点从
`hosts.txt` 里删掉**——保留它，让这个缺口在报告中可见。只有当你确实打算把它排除
出本次审计时才移除该节点。

对于非 root 的 SSH 用户，`cluster_spec_check.sh` 会在远程命令前加 `sudo`，这样
`dmidecode` / `lsblk` / `ethtool` 字段才能保持完整（节点上必须配置好免密
sudo）。
