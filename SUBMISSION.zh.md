**中文** | [English](SUBMISSION.md)

# 提交

要交回什么,以及需要随附填写的表格。

机器生成的部分由 `tools/collect_results.sh` 和 `tools/summarize.py` 产出。
下面的表格是需要人工填写的部分 —— 它们记录脚本无法判断的信息,比如哪些节点是
有意排除的、为什么排除。

---

## 1. 提交包

```bash
bash tools/collect_results.sh --vendor <name> --gpu <model> --dir results
python3 tools/summarize.py results --vendor <name> --gpu <model>
```

提交生成的 `results_<vendor>_<gpu>_<date>.tar.gz`,内含:

```
results/
  SUMMARY.md          自动生成的逐关卡判定
  results.json        同样内容,机器可读
  MANIFEST.txt        哪些存在 / 哪些缺失
  environment.txt     采集主机的硬件 + 软件快照
  preflight/          runtime_*.csv, spec_*.csv
  cublas/             cublas_bench_*.log 和/或 percard_cluster/<node>/dev<N>.log
  nccl/               nccltest_*.log, 1node_nccltest_*.log
  perftest/           perftest_*.log, summary_*.txt, summary_agg_*.txt
```

**没有产出的关卡等于没有测试。** 若某个关卡缺失,要么补跑,要么在下面的备注里
明确说明。`collect_results.sh` 会拒绝打包不完整的集合(除非传 `--allow-partial`),
无论哪种情况 manifest 都会记录缺失项。

### 覆盖范围

| | |
| --- | --- |
| 在范围内的节点数 | |
| 实际测试到的节点数 | |
| 被排除的节点及原因 | |
| 每节点 GPU 数 | |
| GPU 型号 | |
| 测过的 NCCL 规模(节点数) | |
| 使用的 SSH 端口 / 用户 | |

> 被排除的节点必须在此具名列出。未声明的排除,和从未测过的节点无法区分。

---

## 2. 环境验收

填写实测到的值。第 0 关的 CSV 是证据。

| 项目 | 要求 | 实测 | 证据(CSV 字段) |
| --- | --- | --- | --- |
| OS | Ubuntu 24.04 LTS 或同等 | | `os` |
| NVIDIA 驱动 | ≥ 570 | | `driver` |
| CUDA | ≥ 12.9 | | `cuda_api` |
| 用户态 libcuda | 各节点一致,未被 cuda-compat 抢占 | | `libcuda`, `cuda_compat` |
| Docker | ≥ 26 | | `environment.txt` |
| ECC 模式 | 各节点一致 | | `ecc_mode` |
| SM 最高时钟 | 各节点一致 | | `sm_clock_max` |
| 功耗上限 | 各节点一致 | | `power_limit` |
| 计算 HCA | 数量符合预期,全部 ACTIVE | | `compute_hca`, `compute_active` |
| Fabric 类型 | IB 或 RoCE | | `fabric_type` |
| RoCE:GID index、MTU、trust、PFC | 各节点完全相同 | | `roce_gid_v2idx`, `compute_mtu`, `roce_trust`, `roce_pfc_prio` |
| OFED / fabricmanager / peermem | 存在且一致 | | `ofed`, `fabricmanager`, `peermem` |
| 内核 | 各节点完全相同 | | `kernel` |
| GPU-NIC rail 拓扑 | 各节点完全相同 | | `gpu_nic_rail` |
| PCIe ACS | 每个节点都已关闭 | | 见备注 |
| 持久化模式 | 已启用 | | `persistence` |
| memlock | unlimited(第 3 关必需) | | `memlock` |
| 共享存储 | *可选* —— 若使用请注明路径 | | |

**第 0 关结论:** 退出码 `___`(0 = 通过)。关键项差异: `___`

---

## 3. cuBLAS 算力验收

**单机快测(只测 0 号卡 —— 不作为验收依据):**

| | |
| --- | --- |
| 使用的镜像 | |
| `GPU_NAME` | |
| 结果 | 通过 / 失败 |

**全集群逐卡扫描:** `results/cublas/percard_cluster/`,聚合结果为
`fleet_percard.csv`。

| 精度 | 全集群最小 | 中位数 | 最大 | 极差 % | 基线 | 低于基线 ×95 % 的卡数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP4 | | | | | | |
| FP8 | | | | | | |
| INT8 | | | | | | |
| FP16 | | | | | | |
| BF16 | | | | | | |
| TF32 | | | | | | |
| FP32 | | | | | | |

**离群卡**(在一个或多个精度上低于全集群中位数 3 % 以上):

| 节点 | 卡号 | 精度 | 偏差 | 判断 |
| --- | --- | --- | --- | --- |
| | | | | |

若没有请留空。对每张离群卡,说明它是**真实弱卡**(在多个精度上一致偏低)还是
**测量 artifact** —— 并给出支撑该判断的证据。特别注意:H800 的 TF32 读数约为
半速是已知的 cuBLAS artifact;TF32 应改看 torch `allow_tf32`。

**第 1 关结论:** 通过 / 失败 —— `___`

---

## 4. NCCL 互联验收

数值口径为 **16G 消息大小下的峰值 out-of-place busbw(GB/s)**,与
`nccl_bench/baselines.md` 一致。每个测过的规模一张表。

| GPU × 单机卡数 × 机器数 | AllReduce | All2All | AllGather | ReduceScatter | 命中的基线行 |
| --- | ---: | ---: | ---: | ---: | --- |
| ×8×1 | | | | | |
| ×8×2 | | | | | |
| ×8×4 | | | | | |
| ×8×8 | | | | | |

每份日志中的 **Out of bounds values** 必须为 0。任何非零值都是正确性失败 ——
记录并停止:

| 日志 | `# Out of bounds values` |
| --- | --- |
| | |

**判读说明**(不写明这些,数字容易被误读):

- 跨机 All2All 跌到单机数值的 ~10–20 % 是预期行为,不是缺陷。
- H800 AllReduce ≈215 GB/s 是 NVLink 被削减后的硬件规格。
- 若只在同一 rail/leaf 内做了节点对比,请说明 —— 并写明
  `rail_isolation_check.sh` 的结论。

**第 2 关结论:** 通过 / 失败 —— `___`

---

## 5. RDMA 点对点验收

**Rail 隔离**(`rail_isolation_check.sh`):

| | |
| --- | --- |
| 同 rail 带宽 | Gb/s |
| 跨 rail | 通 / 建链失败 |
| 结论 | rail 隔离 / 非 rail 隔离 |

**逐 rail 单 QP**(`summary_*.txt`)—— 200G 端口预期 ~193 Gb/s:

| Rail | write Gb/s | 双向 Gb/s | read Gb/s | write 时延 µs | send 时延 µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| | | | | | |

**逐 rail 聚合**(`summary_agg_*.txt`)—— 取 ≥3 次的 **peak** 判定:

| Rail | peak Gb/s | 预期 peak | 出数的次数 |
| --- | ---: | ---: | --- |
| | | | |

**错误计数器** —— 五个计数器在每条 rail 上、测试前后都必须为 0:

| Rail | out_of_sequence | packet_seq_err | req_cqe_error | local_ack_timeout_err | rnr_nak_retry_err |
| --- | ---: | ---: | ---: | ---: | ---: |
| | | | | | |

任何报为 `NA` 的计数器意味着该 rail 是**未验证**,而不是干净 —— 请说明。

**第 3 关结论:** 通过 / 失败 —— `___`

---

## 6. 签署

| | |
| --- | --- |
| 厂商 | |
| 集群 / 站点 | |
| 日期 | |
| 总体结论(来自 `SUMMARY.md`) | |
| 已知缺口或保留意见 | |
| 联系人 | |

> 如实报告。标明为"结论不明确"的测量是可补救的;一个错误数字被继续用于决策则不是。
> 若某关卡无法执行,请说明是哪一关、为什么,而不是略过不写。
