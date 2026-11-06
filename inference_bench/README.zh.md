# inference_bench — 推理测试模块

[English](README.md) | 中文

推理卡的验收测试。前提是公共前段(节点一致性、单卡 GEMM、NCCL 互联)已通过:弱卡或互联异常会污染这里的结果。

| 子模块 | 测什么 | 状态 |
|---|---|---|
| [`pd_single/`](pd_single/) | 单机 Prefill-only / Decode-only 单测:固定 SLO(TTFT ≤ 7 s、TPOT ≤ 20 ms)下的单 P 容量与单 D 容量,256k 输入、92% prefix 命中,模型 Hy3-FP8 | 草案,协议见 [`pd_single/PROTOCOL.zh.md`](pd_single/PROTOCOL.md)(目前只有中文) |

## 共同约定

- **评测层是 OpenAI 兼容的 HTTP 端点**:厂商在自己的硬件上用 vLLM(上游或基于 vLLM 的 fork/插件)起服务,用本模块提供的**固定客户端**(`vllm==0.24.0` 的 `vllm bench serve` + kit)压测,保证所有厂商发出的请求完全相同。
- **结果包**:每个子模块的 kit 产出 `result.json` 与逐请求原始数据,自检全部通过时 `submission_grade=true`。推理结果的提交方式见仓库根目录 [`SUBMISSION.zh.md`](../SUBMISSION.zh.md) 第 6 节;结果包布局见 `pd_single/PROTOCOL.zh.md` §11。
- **精度**:当前版本不做精度准入,结果标记为 `accuracy.status=unverified`。
