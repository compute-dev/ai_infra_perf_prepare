# inference_bench — inference acceptance tests

English | [中文](README.zh.md)

Acceptance tests for inference cards. Run them only after the common gates (node consistency, single-GPU GEMM, NCCL interconnect) pass — a weak card or a bad link will contaminate the results here.

| Sub-module | What it measures | Status |
|---|---|---|
| [`pd_single/`](pd_single/) | Single-node Prefill-only / Decode-only tests: single-P and single-D capacity under a fixed SLO (TTFT ≤ 7 s, TPOT ≤ 20 ms), 256k input, 92% prefix-cache hit, model Hy3-FP8 | Draft. Protocol: [`pd_single/PROTOCOL.md`](pd_single/PROTOCOL.md) (English; Chinese: `pd_single/PROTOCOL.zh.md`) |

## Common rules

- **The evaluation layer is an OpenAI-compatible HTTP endpoint.** Vendors serve the model on their own hardware with vLLM (upstream, or a vLLM-based fork/plugin) and load it with the **pinned client** shipped here (`vllm bench serve` from `vllm==0.24.0` plus the kit), so every vendor sends byte-identical requests.
- **Result package:** each sub-module's kit writes `result.json` plus per-request raw data; `submission_grade=true` only when all self-checks pass. How to submit inference results is section 6 of the repo-root [`SUBMISSION.md`](../SUBMISSION.md); the result-package layout is in `pd_single/PROTOCOL.md` §11.
- **Accuracy:** this version has no accuracy gate; results are marked `accuracy.status=unverified`.
