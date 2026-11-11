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
| 是否跑了推理关(仅推理卡) | |
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

## 6. 推理验收(仅推理卡)

只适用于推理卡,且必须在 Gate 0~3 验收通过之后做:弱卡或坏链路会污染这里的数字。
一份提交覆盖**一种硬件型号 + 一套引擎配置**。

> **状态:草案,且不在打包脚本覆盖范围内。** `tools/collect_results.sh` 与
> `tools/summarize.py` 还不认识这一关 —— `results/inference/` 需要你自己手工打包,
> 与它们产出的 tarball 一并提交。协议本身是草案
> ([`inference_bench/pd_single/PROTOCOL.zh.md`](inference_bench/pd_single/PROTOCOL.zh.md)),
> 待定项见该文档 §12。

测什么(依据 [`inference_bench/pd_single/`](inference_bench/pd_single/)):固定 SLO
(P 的 mean TTFT ≤ 7 s、D 的 mean TPOT ≤ 20 ms)下的单机 **Prefill-only** 与
**Decode-only** 容量,256k 输入、92% prefix cache 命中、模型 Hy3-FP8。
端到端 PD、KV 传输、多机**都不在范围内**。

| # | 项 | 要求值 / 证据 |
| --- | --- | --- |
| 1 | `submission_grade` | 两份 `result.json` 都为 `true`。`false` 表示有自检未通过,应修正后重跑,不要直接提交 |
| 2 | 清单 | `vendor_manifest.yaml` 每项都已填写;`validate-manifest` 退出码 0 |
| 3 | 数据集指纹 | `e9beb87e24e1`;`--repeat>1` 时另需 `9adcdac59475`。本地用 `gen-datasets` 生成,由 kit 校验(C2) |
| 4 | tokenizer 指纹 | `7f062efabe6a` —— Hugging Face `tencent/Hy3-FP8` 的三个文件(C2t) |
| 5 | 客户端 | 固定镜像,`vllm==0.24.0`;请求体哈希与 golden 一致(C13)。不得替换为自有客户端 |
| 6 | 重复 | 每个并发档 ≥2 次独立运行,并报离散度。离散度超过 3% 阻断提交(C9) |
| 7 | 单 P 容量 | `capacity.max_tpm` —— 所有 PASS 档中 `total_tpm` 最大的一档(主口径);报 `total_tpm`、`per_gpu_total_tpm`、QPM(实测与稳态)与对应并发档,并同时给出 `capacity.max_qpm` |
| 8 | 单 D 容量 | `capacity.contiguous` —— 从最小档起连续 PASS 的最大档(D 主指标);报 `total_tpm`、`output_tpm`、`per_gpu_total_tpm`、QPM(实测与稳态)与对应并发档,并同时给出 `capacity.max_tpm` 与 `capacity.max_qpm` |
| 8b | 单卡 TPM 分母 | 注明分母取 TP 度还是整机卡数。kit 按清单声明的 `serve.tp` 相除,并记录在 `capacity.per_gpu_basis` |
| 9 | 拐点证据 | 最后一个 PASS 档与第一个 FAIL 档,都带数值。`capacity.knee_found` 为 false(无档破线)时,写成「容量 ≥ `capacity.max_tier_measured`,拐点未测出」,**不能**当作容量 |
| 10 | 精度 | 本版 `accuracy.status` 为 `unverified`。需如实写明容量数字未经质量把关 |
| 11 | 服务日志 | 两个端点各自的启动日志,含 `GPU KV cache size`(C11) |
| 12 | 抢占 | 逐档 `preemptions` 必须为 0(C15)。发生抢占的档,其延迟数字不可用 |

**档位上限**:随仓库发布的数据集下,D 的并发档上限为 c=20、P 为 c=8 —— 这由数据集规模决定,
与你的硬件无关。kit 会在开跑前拒绝超限档位。若你的容量超过该上限,请在签收节写明:
不做协议级变更就测不出拐点。

### 结果目录布局

```
results/
  inference/
    pd_single/
      p_only/
        result.json          P 角色的主结果
        summary.csv          每并发档一行
        c<N>_r<k>.json/.log  逐档逐次原始数据
      d_only/
        result.json          D 角色的主结果
        summary.csv
        c<N>_r<k>.json/.log
      vendor_manifest.yaml   本次运行所用清单
      serve_p.log            P 端点启动日志
      serve_d.log            D 端点启动日志
```

两个角色分目录存放:一次 `run` 只产出一个角色。没有输出的角色即未测,
应在签收节写明,不要静默省略。

### 容量表

| | 并发档 | 总 TPM | 单卡 TPM | 实测 QPM | 稳态 QPM | 是否测到拐点 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 单 P 容量(`max_tpm`) | | | | | | | |
| 单 D 容量(`contiguous`) | | | | | | | |
| 单 D SLO 内最大(`max_tpm`) | | | | | | | |

单卡 TPM 的分母(TP 度或整机卡数): `___`

**推理关结果:** pass / fail — `___`

---

## 7. 签署

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
