# pdbench 接入协议与结果 Schema(草案 v0.5)

> 状态:**设计草案**。§14 列出已在参考平台验证的内容和未验证的部分;标注「待定」「待验证」的条目仍无结论。
> 目的:一套让各家硬件厂商在自己的服务端上自测单机 Prefill / Decode 容量、评测方可复测的协议与工具(kit)。
> 本仓库只含协议、kit、客户端镜像定义与参考适配;评测方的汇总评估、复测流程不在本仓库。

英文版:[`PROTOCOL.md`](PROTOCOL.md)。两版不一致时**以本文为准**。

## 1. 目标与非目标

**目标**

- 任一厂商在自己的机器上,按本协议跑出「单 P 容量」「单 D 容量」,产出一份自洽、可校验的结果包。
- 评测方把多家结果放进同一张表做跨硬件对比,并对可疑数字做复测。
- 厂商接入只需要:**起两个 OpenAI 兼容端点 + 填一份清单 + 跑我们的 kit**。不需要给评测方机器权限,服务端不需要使用评测方的镜像。

**非目标**

- 不测真实 PD 端到端(交接开销、排队、KV 传输)。
- 不测多机互联、集群扩展性。
- 不替代公共前段(单卡算力、NCCL 等)。

## 2. 分层与信任模型

| 层 | 谁负责 | 内容 |
|---|---|---|
| 端点 | 厂商 | 用 vLLM(上游或厂商 fork/插件)起 P、D 两个实例,如实声明配置 |
| 协议常量 | kit | SLO、场景、并发档、请求数、温度、seed、冷启动规则 |
| 数据 | kit | 数据集生成器、tokenizer 校验、指纹 |
| 加载与采集 | kit | 固定版本 `vllm bench serve` + 指标采集 |
| 自检 | kit | 阻断性校验(见 §8),不过则不出结果 |
| 评估与发布 | 评测方 | 多厂商对比、与参考平台对照、复测、报告(不在本仓库) |

**信任模型:厂商自测 + 我们复测。**

- 厂商侧:kit 自动拦截老实误配(请求数不足、命中率不达标、指纹不符、输出长度没生效)。
- 评测方侧:用评测方持有的**另一套种子集**复测。复测值与自报值偏差 > ±5% 时,先要求厂商解释;解释不成立则以复测值为准(§9)。
- kit 不承诺能防住刻意作假。对于刻意针对固定输入做的优化(预热、记忆前缀),靠另一套种子集暴露。

**前提决策(已定)**

1. 被测硬件包含非 NVIDIA → 协议定在 OpenAI 兼容 HTTP 端点层,不假设 CUDA。
2. 引擎固定为 vLLM(上游或基于 vLLM 的 fork/插件)。非 vLLM 栈不在本期范围。
3. 数据集是合成的,可公开:内容是随机占位 token;**每条输入长度相同**(同一段共享前缀 + 定长唯一后缀);**输出长度也固定**——P 单测每条 1 个 token,D 单测每条 1,200 个 token(§4)。生成过程不读取任何真实数据,数据集里没有真实内容,也没有真实请求的长度。
4. 精度:先试官方 Hy3-FP8;跑不了才允许自选,但必须声明。**v1 不设精度准入门槛**(见 §7):结果包标注「精度未验证」,精度对齐集后置。

## 3. 被测接口(端点契约)

厂商需提供两个**互斥使用**的端点(单测一次只压一个角色,不需要 P/D 同时在线):

| 角色 | 必需能力 |
|---|---|
| P(prefill 单测) | `POST /v1/chat/completions`;支持 `max_completion_tokens`;`ignore_eos` 生效;prefix cache 开 |
| D(decode 单测) | 同上,且 **prefix cache 命中率可测** |

**命中率怎么测(按优先级)**

1. Prometheus `/metrics` 的 `prefix_cache_hits_total` / `prefix_cache_queries_total` 差值(vLLM 系都有;参考平台用的就是这条)。
2. 响应 `usage.prompt_tokens_details.cached_tokens` 累加。
3. 两者都不可用 → **D 单测结果无效**(无法证明只做了 decode)。

厂商须在清单里声明用了哪一条,kit 按声明校验;声明与实测不符即失败。

**启动要求**

- 单测期间机器独占,不能有别的 GPU 负载。
- 每次运行前清理编译缓存(冷编译起点一致);kit 记录是否清理。
- P、D 实例**不需要** KV 传输 connector。在参考平台上验证过去掉 `NixlConnector` 后 P/D 指标无变化(§14),厂商不需要装 NIXL/UCX、不需要配数据面网卡。

**清单 `vendor_manifest.yaml`(必填)**

```yaml
protocol_version: pdbench-1.0
vendor: <厂商>
run_id: <厂商自定,唯一>
hardware:
  gpu_model: <型号>
  gpu_count: <N>            # 单机卡数
  gpu_mem_gib: <N>
  cpu_model: <型号>
  cpu_cores: <N>
  ram_gib: <N>
  driver: <版本>
  cuda: <版本>
image:
  ref: <镜像地址>            # 或 docker save 的 sha256
  digest: <sha256>
engine:
  vllm_version: <版本>
  fork: <仓库地址/commit 或 "upstream">
  deviations: [<算子替换/自定义 attention/调度器改动/未合并补丁……>]
precision:
  weights: <fp8 | bf16 | int8 | ...>
  kv_cache: <fp8_e4m3 | auto | ...>
  rationale: <若 weights != fp8,说明为什么官方 Hy3-FP8 跑不了>
endpoints:
  p: http://<host>:<port>
  d: http://<host>:<port>
  metrics: http://<host>:<port>/metrics
  hit_rate_source: metrics | usage   # 见 §3
serve:
  p_cmd: <完整启动命令,含全部环境变量>
  d_cmd: <完整启动命令>
  tp: <N>
  max_model_len: <N>
  gpu_mem_util: <0.x>
  prefix_caching: true               # D 必需
  chunked_prefill: <true/false/默认>
  mtp_tokens: <0 | N>
  cudagraph: <描述>
  other_optimizations: [<...>]
client:
  image: <kit 提供,厂商不得替换>
  cpuset: <预留 CPU 核,见 §12-O2>
```

## 4. 固定协议常量(换硬件不改)

| 项 | 值 |
|---|---|
| `protocol_version` | `pdbench-1.0` |
| 场景 | 256k:**每条输入长度相同**,设计值 84,992 token = 共享前缀 78,192 + 定长唯一后缀 6,800(设计命中 0.92);实际 token 数因边界对齐与特殊 token 略少于设计值,C6 按 ±2% 校验 |
| 输出 | **固定长度**:P 单测每条 1 个 token;D 单测回放遍每条 **1,200** 个 token(`--custom-output-len 1200`),`ignore_eos=true`。数据集文件里仍有一列 `output_tokens`(生成器按对数正态写入的历史字段),**kit 不使用它**;保留它是为了不改变数据集指纹。为什么不用变长输出见 §13 |
| SLO | mean TTFT ≤ 7,000 ms(P);mean TPOT ≤ 20 ms(D)。p99 并列报告,不参与主判定 |
| 温度 | **显式 `temperature=0.9`(P、D 都发)**;`seed=0` |
| 请求速率 | `request_rate=inf`(闭环) |
| 并发档 | kit 默认 P:1~8;D:1,2,4,6,8,10,12,16,20(上限 c=20,见 §6 并发档上限),配合 `--stop-after-fail 2` 连续两档破线即停。`first_fail_concurrency` 随所测档位而定,**容量判定不受影响** |
| 请求数 | P:每档 80(≥10×c 要求 c≤8);D:每档 `min(200, max(40, 10×c))` |
| D 灌缓存 | 回放前先跑一遍 `N` 条(`max_completion_tokens=1`,c=4),`N = min(回放集条数, max(40, 10×最高档))`,之后每档回放 |
| warmup | P 每档 10;D 每档 2(不计入统计) |
| 重复 | 每个单元格 **≥2 次独立运行**(厂商自测为同一实例上两次;跨机器重复见 §14 未覆盖项),报告离散度 |
| 容量定义 | **所测档位全部 PASS 时 `capacity.knee_found` 为 false,结果只能读作「容量 ≥ 所测最高档,拐点未测出」,不得当作该硬件的容量。** P:所有 PASS 档中的最大 QPM;D:①从最小档起连续 PASS 的最大档 QPM(主指标)②所有 PASS 档最大 QPM;两者都报「实测」与「稳态」(`c×60000/mean_e2el_ms`) |

**温度说明**:P/D 都显式发 `temperature=0.9`,不依赖服务端 `generation_config` 的默认值(各引擎/镜像默认值不保证一致)。在参考平台上验证过显式发送对 P 单测指标无影响(§14)。

## 5. P 单测

1. 起 P 端点,等就绪;记录 `GPU KV cache size` 与全部启动参数。
2. 逐档:采样 `/metrics` 命中计数 → `vllm bench serve`(数据集第 c 档文件,`--custom-output-len 1`、`--num-prompts 80`、`--max-concurrency c`、`--num-warmups 10`、`--temperature 0.9`、`--seed 0`、`--ignore-eos`)→ 再采样命中计数。
3. 记录:mean/p50/p99 TTFT、QPM、稳态 QPM、命中率(计数差)、输入 token 均值、完成数。
4. 判定:mean TTFT ≤ 7 s;容量 = PASS 档最大 QPM。

## 6. D 单测(两遍法)

**为什么两遍**:单测 D 不能依赖 P 侧 KV 传输;而假 KV(`DecodeBenchConnector`)会让 MTP 接受率塌到 2.5%,不可用。做法是让 D 自己把 KV 灌进 prefix cache:同一批 prompt 先以 `max_completion_tokens=1` 走一遍(只做 prefill,缓存驻留),再逐档回放——回放时近 100% 命中,基本只做 decode。

1. 起 D 端点(prefix cache 必须开)。
2. 灌缓存:`N` 条,c=4,`max_completion_tokens=1`;记录本遍命中率。**`N` 按本次扫描的目标档位算** —— `N = min(回放集条数, max(40, 10×最高档))`,不把回放集顶满(见下「为什么 N 不顶满」)。
3. **逐档重新访问**:每档回放前,先对该档要用的 prompt 以 `max_completion_tokens=1`、c=4 走一遍,记录本遍命中率(`tiers[].touch_hit`)。仍在缓存则开销接近零;已被逐出则重新算回并回到最近使用位置。**该遍在测量窗口之外,不计入该档指标。**
4. 逐档回放:同一数据集、同 seed,`--custom-output-len 1200`(每条固定输出 1,200 token)、请求数按 §4 公式。
5. 每档记录:mean/p99 TPOT、TTFT、QPM、稳态 QPM(固定输出长度下两者应基本重合;差得多说明收尾阶段或服务端排队有异常)、命中率、`touch_hit`、MTP 接受率(若开 MTP)、最大 running/waiting、最大 KV 使用率。
6. 判定:mean TPOT ≤ 20 ms。

**为什么要第 3 步**:只在开头灌一次缓存时,尚未被任何档使用的 prompt 自灌缓存后再未被访问,在 LRU 顺序里最旧;而每条请求的 1,200 个输出 token 同样占用 KV,累计量随档位增长。累计量接近 KV 容量后,这些 prompt 先被逐出,表现为**该档第 1 次运行命中率下降、耗时偏高,第 2 次恢复**,两次重复发散。参考平台实测:两种配置分别从 c=10、c=14 起出现,两台机器起始档一致,两次重复差约 5%~9%(该档 TPOT 与 QPM)。`touch_hit < 1` 即该档开跑前发生过逐出。**显存更小的硬件会在更低档位触发。**

**前提(容量规则)**:灌缓存的驻留量 ≈ `共享前缀 78,192 + N × 6,800`(共享前缀只存一份;`N` 为实际灌入条数,见 §6 第 2 步)。厂商须从启动日志声明 `GPU KV cache size`,kit 校验:

```
容量 ≥ (78,192 + N×6,800 + c_max×6,800) × 1.05
```

**该式只算 prompt 的驻留量,不含每条请求 1,200 个输出 token 的累计占用** —— 后者随档位增长,由 §6 第 3 步的逐档重新访问兜住,不改变本式。算例:最高档 20 → N=200 → 要求 ≥ 1,652,902 token;最高档 8 → N=80 → 要求 ≥ 710,422 token。容量不足时按该式反解 N 降档并写入结果包(`fill.degraded=true`);若 N < 单档请求数,则该档 D 结果标记「受限」不参与容量判定。**该规则是解析推导;在参考平台上灌入遍命中率≈共享前缀占比、回放遍≈100%,与推导一致,但小显存卡未实测。**

**并发档上限**:单档请求数被回放集条数封顶 `min(FILL_N, 回放集行数)`,而 C4 要求每档请求数 ≥ 10×并发。两者在 `10×c > 可用条数` 时无法同时满足,该档必然 C4 FAIL。因此当前回放集(200 条)下 **D 的并发档上限为 c=20**,kit 在开跑前就拒绝超限档位(退出码 2),默认 D 档位为 `1 2 4 6 8 10 12 16 20`。**若被测硬件的 D 容量本身 ≥ 20,当前回放集测不出它的拐点** —— 需先加大 FILL_N 并重新生成更大的回放集,这会改变数据集指纹,属协议级变更。

**为什么 N 不顶满**:KV 里同时住着两类数据 —— 回放集常驻(`共享前缀 + N×后缀`)与**历史请求的输出块**(每条 1,200 token,随档位累计,容量式不计入)。
两者之和越过 KV 容量后,服务端按 LRU 逐出,最先被逐出的是本轮还没轮到的回放集 prompt,于是该档的重新访问要真的重新 prefill(只多花时间,不影响结果 —— 见 §6 第 3 步)。
N 越大,常驻占比越高,逐出从越低的档开始。参考平台实测(最高档 20):N=200 时回放集常驻约占 KV 的 67%,逐出从 c=14 起;按同一容量估算,N=250 占 83%(逐出从 c=12 起)、N=300 占 99%(从 c=2 起)。
**而扫描结束时的累计量必然超过 KV(实测约 2.2 倍),所以逐出无法靠调小 N 消除** —— N 的取法只决定要重新 prefill 多少档。
因此 N 按目标档位取够用即可:`c_max = N/10`,测到拐点之外没有意义。拐点未知时先用小 N 粗扫圈定,再按圈定结果定 N 做细扫。

## 7. 精度规则

1. **先尝试官方 Hy3-FP8 权重**。厂商硬件/内核只要跑得起来,就必须用它,不做取舍。
2. 跑不起来,允许自选精度,但必须:
   - 在清单里声明 weights / KV cache / 激活量化方案与原因;
   - 结果表里带精度标签,**不与 FP8 结果混在一列比**。
3. 自选精度的容量数字只作参考,不作为该硬件「FP8 等价容量」对外表述。
4. **v1 不做精度准入**:所有结果包的 `accuracy.status` 固定为 `unverified`,报告里如实写「容量数字未经质量把关」。原因:压测用随机占位 token,输出无意义,**性能数据本身看不出输出对不对**,所以这是已知缺口,不是已排除的风险(参考平台自身的 fp8 KV 精度也未验证)。
5. **冒烟档(可选,kit 提供)**:约 50 条 prompt、greedy,只检查输出非乱码/不重复/能正常结束,结果写入 `accuracy.smoke`;抓不到细微精度损失。
6. 升级触发:第一家厂商接入、或数字要用于采购/选型判断时,升级到「标准档」(现成工具跑公开题集子集,与参考平台的 FP8 分数对比)。

## 8. kit 自动自检(阻断项)

C2t、C13、C14 是**前置检查**:`run` 开跑前先做,不过直接退出(`--continue-on-preflight-fail` 仅供调试)。其余检查在跑完后判定。任一 FAIL → `run` 退出码 1,`result.json` 仍写出但 `submission_grade=false`(便于排查);退出码只看 FAIL;`submission_grade=true` 则要求提供清单且**每一项都是 PASS 或 WARN**(出现 SKIP,如没给 `--server-log`,即为 false)。

| # | 检查 | 阈值 |
|---|---|---|
| C1 | 清单 `protocol_version` 与 kit 一致;kit 自身 sha256 记入结果。**不核对** SLO/场景/温度/seed(这些是 kit 内置常量,不由厂商提供) | 无清单 → SKIP |
| C2 | 9 个基础数据集文件的汇总指纹 = `e9beb87e24e1`(`--repeat>1` 时另核重复集指纹 `9adcdac59475`)。**只核汇总指纹**,不逐文件报告 | 精确匹配 |
| C2t | tokenizer 三文件(`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`)汇总指纹 = `7f062efabe6a`(与 Hugging Face `tencent/Hy3-FP8` 官方发布的三个文件逐字节一致)。chat_template 决定服务端实际 prompt | 精确匹配 |
| C13 | 客户端自检:用真实客户端命令行打本地 stub,收到的请求体哈希集合 = golden(`kit/selftest/golden_requests.json`);且客户端 `vllm --version` = 固定版本 `0.24.0` | 精确匹配 |
| C14 | 服务端探测:发一条与真实客户端同形(`content` 为分段列表)的小流式请求,流里必须带完整 `usage`,且要 8 个输出 token 得 8 个(`ignore_eos` 生效)。客户端的输入 token 数取自服务端 `usage.prompt_tokens`,缺失时会静默回退成客户端自己数的值,C6 随之失效,所以必须单独校验 | 通过 |
| C15 | 每次运行的服务端抢占计数(`vllm:num_preemptions_total` 增量)必须为 0 —— KV 吃紧时服务端会抢占运行中的请求并重算,这会抬高该档延迟而命中率看不出来 | 取不到该指标 → SKIP |
| C3 | 每次运行完成数 = 计划数且无错误 | 100% |
| C4 | 每档请求数 ≥ 10×并发 | — |
| C5 | 实际命中率(/metrics 计数差):P 每次 ∈ [0.90, 0.95](上限拦"后缀被缓存");D 回放每次 ≥ 0.999(下限拦"回放集被逐出";由 0.98 提高 —— 旧下限会放过被逐出后重新 prefill 的运行) | 见左 |
| C6 | 每次输入 token 均值 84,992 ±2% | 见左 |
| C7 | P:逐请求输出长度均为 1。D:逐请求输出长度均为 1,200(`ignore_eos` 生效、没有被截断;拿不到逐请求长度也判 FAIL)。C7f:灌缓存全部完成 | 精确匹配 |
| C8 | 客户端 warmup 条数与协议一致(从客户端日志核对);冷编译缓存已清、机器独占**只由清单声明**,kit 不能核实 | 无清单 → FAIL |
| C9 | 每档 ≥2 次运行;TTFT/TPOT 与 QPM 的重复相对差 ≤1% 通过,1%~3% WARN,>3% FAIL;只跑 1 次 → FAIL(无清单时 SKIP) | 阈值依据参考平台同实例重复的实测离散度 |
| C10 | 声明 `hit_rate_source=metrics` 且 /metrics 实际可采集。`usage` 来源 kit 未实现 → SKIP | 无清单 → SKIP |
| C11 | 提供 `--server-log` 且日志含 `GPU KV cache size`。**只核对"有这一行",不核对启动命令与清单一致** | 无日志 → SKIP |
| C12 | D:KV 容量 ≥ (78192 + FILL_N×6800 + c_max×6800)×1.05(§6) | 无 KV 数据 → SKIP |

## 9. 复测与判定

- 复测条件:全部厂商结果中抽检(建议至少 1 家/批次,重点抽「异常好」的);或 C9 离散度超限;或厂商自选精度(精度未验证,复测是唯一的外部校验)。
- 复测方式:同一台机器、同一镜像、同一启动命令,**私有种子集**(同分布、不同 `offset-base`),协议不变。
- 判定口径:比较**单 P 容量 QPM**、**单 D 容量 QPM(连续达标)** 两个数。偏差 = |复测 − 自报| / 自报。
- 偏差 ≤ 5%:接受自报值,报告注明已复测。
- 偏差 > 5%:要求厂商解释。解释成立的偏差(如温度/频率/邻居干扰)记录在案;不成立则以复测值为准,并在报告中标注。
- 复测值永远单独存档,不覆盖厂商自报值。

## 10. 本仓库范围

本仓库包含:协议(本文)、kit(`kit/`)、客户端镜像定义(`client/`)、清单模板与参考适配(`templates/`、`examples/`)。

**不包含**:评测方对多家结果的汇总评估、复测流程、评测方持有的另一套种子集,以及任何厂商的结果。厂商只需要本仓库的内容就能完成自测并产出结果包。

## 11. 目录结构与使用方式

```
pd_single/                           # 本目录:推理测试模块 inference_bench/ 的子模块「单机 P/D 单测」
├── PROTOCOL.zh.md                   # 本文(中文,两版不一致时以此为准)
├── PROTOCOL.md                      # 英文版
├── kit/
│   ├── pdbench.py                   # gen-datasets / validate-manifest / run(P、D 单测 + 自检 + 结果包)
│   ├── dataset/                     # gen_prefix_dataset.py、gen_dataset.py(从 inference/dataset/ 复制)
│   ├── selftest/                    # 客户端自检:prompts.jsonl(3 条固定 prompt)、golden_requests.json(请求体哈希)
│   └── tests/test_checks.py         # 自检回归用例
├── client/Dockerfile                # 固定客户端镜像(验证范围见 §14)
├── templates/vendor_manifest.yaml   # 厂商清单模板(未填项会被 validate-manifest 拒绝)
└── examples/h20_vllm/               # 参考适配:h20_manifest.yaml、serve_p.sh、serve_d.sh
```

**尚未建**:`schema/result.schema.json`(下方骨架仅作示意)。客户端版本固定为公开发布版 `vllm==0.24.0`;`client/Dockerfile` 的验证范围见 §14。

```bash
python3 -I kit/pdbench.py gen-datasets --tokenizer <模型目录> --out-dir <数据集目录>      # 生成并校验指纹(含重复集)
python3 -I kit/pdbench.py validate-manifest vendor_manifest.yaml
python3 -I kit/pdbench.py selftest-client --model-path <模型目录>      # 开测前先跑:核对客户端请求体(无需被测服务)
python3 -I kit/pdbench.py run --role p --base-url http://<host>:<port> --model-path <模型目录> \
    --served-model-name <名> --dataset-dir <数据集目录> --out <输出目录> --manifest vendor_manifest.yaml --server-log <P 服务日志>
python3 -I kit/pdbench.py run --role d ... --stop-after-fail 2     # 默认 --repeat 2;--tiers 不填用默认档位
```

运行 kit 需要 Python ≥ 3.8(更旧的版本会直接报错并提示改用客户端镜像);客户端镜像的构建与运行要点见 `client/Dockerfile` 头部注释。

自检不过时进程退出码为 1,`result.json` 里 `submission_grade=false`;`run` 同时写 `summary.csv`(每并发档一行,含重复均值)。

**结果包**:`result.json`(主)+ `summary.csv` + 每档原始 `c<N>_r<k>.json/.log`(含逐请求 `ttfts/itls/output_lens`,默认不含 `generated_texts`)。汇总评估读 `result.json`,不在本仓库内。

### result.json 骨架

```json
{
  "protocol_version": "pdbench-1.0",
  "run_id": "...",
  "manifest_sha256": "...",
  "created_at": "YYYY-MM-DDThh:mm:ss",
  "dataset": { "fingerprint": "e9beb87e24e1" },
  "slo": { "ttft_ms": 7000, "tpot_ms": 20 },
  "p_only": {
    "tiers": [{
      "concurrency": 4, "num_prompts": 80, "completed": 80,
      "mean_input_tokens": 84800, "mean_output_tokens": 1,
      "mean_ttft_ms": 1111.1, "p50_ttft_ms": 1111.1, "p99_ttft_ms": 2222.2,
      "qpm": 11.1, "steady_qpm": 11.1, "prefix_hit": 0.93,
      "verdict": "PASS", "repeat": [{"qpm": 11.1, "mean_ttft_ms": 1111.1, "p99_ttft_ms": 2222.2}, {"...": "..."}]
    }],
    "capacity": { "max_qpm": { "concurrency": 4, "qpm": 11.1, "steady_qpm": 11.1 }, "first_fail_concurrency": 5,
                  "knee_found": true, "max_tier_measured": 8 }
  },
  "d_only": {
    "fill": { "num_prompts": 200, "completed": 200, "hit_ratio": 0.92 },
    "tiers": [{
      "concurrency": 8, "num_prompts": 80, "completed": 80,
      "mean_input_tokens": 84800, "mean_output_tokens": 1200,
      "mean_ttft_ms": 333.3, "mean_tpot_ms": 9.99, "p99_tpot_ms": 9.99,
      "qpm": 22.2, "steady_qpm": 22.2, "prefix_hit": 1.0, "touch_hit": 1.0, "preemptions": 0,
      "mtp_accept": null, "verdict": "PASS", "repeat": [{"...": "..."}]
    }],
    "capacity": {
      "contiguous": { "concurrency": 8, "qpm": 22.2, "steady_qpm": 22.2 },
      "max_qpm": { "concurrency": 8, "qpm": 22.2, "steady_qpm": 22.2 }, "first_fail_concurrency": 10,
      "knee_found": true, "max_tier_measured": 12
    }
  },
  "accuracy": { "status": "unverified", "smoke": null },
  "checks": [ { "id": "C2", "status": "PASS", "detail": "数据集指纹 e9beb87e24e1" } ]
}
```

> 上面的数字是**虚构的占位值**(11.1、22.2 之类),不代表任何硬件的实测结果。实际 `run` 每个角色各写一个 `result.json`,字段以 kit 输出为准。

## 12. 待定 / 待验证

| # | 项 | 说明与建议 |
|---|---|---|
| O1 | 精度对齐集(**后续**) | v1 不做准入(§7)。后续:冒烟档(半天)→ 标准档(lm-eval 子集 + 与参考平台 FP8 分数对比,1~2 天)→ 完整档(含长上下文,一周以上,难点在定阈值)。 |
| O2 | 客户端资源隔离未验证 | 现在客户端与服务同机同容器,抢 CPU 的影响没测过。建议固定预留若干 vCPU 并在清单 `client.cpuset` 声明;先做「同机 vs 隔离」对照。 |
| O4 | 灌缓存降档 | 规则已定(§6 容量式);kit 只校验、**不自动降档**;小显存卡未实测。 |
| O4b | 并发档上限 | 回放集 200 条 ⇒ D 档位上限 c=20(§6 并发档上限)。kit 开跑前拒绝超限档位,但**不自动加大回放集**;D 容量 ≥20 的硬件需要先做协议级变更(更大的 FILL_N 与回放集、新指纹)。 |
| O6 | 能力缺失的处理 | 不支持 `max_completion_tokens` / `ignore_eos` / prefix cache 的栈 → 判「不适用」并说明,不出容量数字。 |
| ~~O7~~ | ~~数据集分发方式~~ | **已定**:发生成器 + 指纹,厂商本地生成后由 kit 校验(数据集打包约 1 GB,不随仓库分发)。Hy3 是 Apache-2.0 开源模型(Hugging Face `tencent/Hy3-FP8`),tokenizer 允许随 kit 分发;但 kit **不内置**它——厂商起服务本来就要下载完整模型目录,`gen-datasets --tokenizer` 与 C2t 直接用该目录,避免仓库里再存一份可能过时的拷贝。 |
| O8 | 附加配置怎么进报表 | 主表只放主配置(无 MTP、无投机);MTP / 其他加速作为「附加配置」单列,不混入主对比。 |
| O12 | 实现缺口 | `usage.cached_tokens` 命中率来源、FILL_N 自动降档、结果包打包、`result.json` 的正式 JSON Schema、跨机器重复。 |
| O14 | 服务端对请求体的兼容性 | 客户端发的 `content` 是分段列表(见本节下方的请求体),不同引擎/chat template 对此的处理未知;C14 探测能暴露「直接拒绝」,**暴露不了「接受但切出不同 prompt」**——后者只能靠 C6(服务端 `prompt_tokens`)间接发现。 |

**客户端请求体**(vLLM `0.24.0` 与更早的开发构建经 stub 抓包确认一致):

```json
{"model": "<served-model-name>", "messages": [{"role": "user", "content": [{"type": "text", "text": "<prompt>"}]}],
 "max_completion_tokens": <N>, "stream": true, "stream_options": {"include_usage": true}, "ignore_eos": true, "temperature": 0.9}
```

注意 `content` 是**分段列表**而不是字符串。客户端的输入 token 数取自服务端流里的 `usage.prompt_tokens`;服务端不带 `usage` 时客户端会静默回退成自己数的值,C6 随之失效,所以才有 C14。

## 13. 已排除的做法(别再试)

- `DecodeBenchConnector` 假 KV:会让 MTP 接受率塌到不可用的水平。
- 每档请求数过少(如固定 60 条):会明显低估 QPM(收尾阶段并发不满),已改为 ≥10×并发。
- 只报 mean 不报 p99:两者给出的容量可能明显不同,两者都要报。
- 手抄起服务命令进报告:会与脚本 drift,清单里的命令应由配置程序化生成。
- 单测给 P/D 起 KV 传输 connector(§14:在参考平台上无影响)。
- D 单测用变长输出(逐条读数据集的 `output_tokens`,对数正态分布,均值 1,200):① 每档只取回放集的前 N 条,各档实际平均输出长度不同(参考平台上在约 ±3% 范围内来回变),QPM 跟着漂移,相邻档不可比;② 压测快结束时,剩下的长请求在低并发下单独跑,拉长总时长,**稳态 QPM 比实测 QPM 高 7%~13%**(即实测被低估);③ mean TPOT 按条平均,短请求和长请求权重相同。改为固定 1,200 后,在参考平台上对照(c=4/6/8,两台机器,各重复 2 次):mean TPOT 差异在 1% 以内、SLO 判定与容量档不变,p99 TPOT 略低;稳态与实测 QPM 的差距从 7%~13% 降到 0.2% 以内。
- P 的重复运行复用同一份数据集:第二遍的唯一后缀已在 prefix cache 里,TTFT 会掉到几分之一,等于没算 prefill(kit 用独立的 `_r2/_r3` 数据集并给命中率设上限)。

## 14. 验证状态

**已在参考平台验证**(单机 8 卡、Hy3-FP8、vLLM 上游系;具体数字不公开):

- kit 的 P(8 档)、D(多档)全档位运行与评测方自有基线逐档一致;`submission_grade` 路径、前置检查、自检回归用例(`kit/tests/`)通过。
- D 改为固定输出 1,200 后,在参考平台上做了对照(§13):TPOT 与 SLO 判定不变,实测 QPM 不再被收尾拖低。随后按固定输出口径在参考平台上补测了 **D 三种配置的全档位 + P 三种配置**,共 9 份结果全部 `submission_grade=true`,其中每种 D 配置都在两台机器上各跑一次。
- 固定客户端 `vllm==0.24.0` 发出的请求体与更早的开发构建逐字节等价(stub 抓包哈希一致);两者指标对账一致。
- 去掉 KV 传输 connector、P 显式 `temperature=0.9`,对 P/D 指标无影响。
- `client/Dockerfile` 在两台机器上构建成功,镜像内 `selftest-client` 通过。

**未验证 / 未覆盖**(使用前请知悉):

- **D 的并发档上限 c=20、P 的并发档上限 c=8**(数据集规模所致,见 §6 并发档上限与 O4b):容量超过该档位的硬件测不出拐点,kit 会在开跑前拒绝超限档位。

- 只验证了 vLLM 上游系的服务端,**没有非 NVIDIA 硬件、没有非 vLLM 的服务端、没有小显存卡**。
- 重复运行是同一实例上的两次;跨机器重复由「同配置在两台机器各跑一次」提供,但两台不同时跑同一档,**不能分离机器间差异与时段漂移**;C9 只抓同实例的随机波动。
- `usage.cached_tokens` 命中率来源、`FILL_N` 自动降档未实现。
- `client/Dockerfile` 的默认基础镜像 `python:3.12-slim` 未验证(验证用的是等价的 Python 3.12 镜像);旧版 Docker(实测 19.03)需要 `--security-opt seccomp=unconfined`,见 Dockerfile 注释。
- 请求体等价只证明**请求相同**,不证明厂商服务端对它的处理相同(O14)。
- **没有做精度校验**(§7),所有容量数字都未经质量把关。
