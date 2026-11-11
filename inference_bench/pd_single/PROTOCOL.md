# pdbench — onboarding protocol and result schema (draft v0.5)

> Status: **design draft**. §14 lists what has been validated on the reference platform and what has not; items marked "open" or "unvalidated" have no conclusion yet.
> Purpose: one protocol plus one tool (the kit) that lets any hardware vendor measure single-node Prefill / Decode capacity on their own server, and lets the evaluator re-measure it.
> This repository contains only the protocol, the kit, the client image definition and a reference adaptation. The evaluator's cross-vendor evaluation and re-measurement workflow are not in this repository.

Chinese version: [`PROTOCOL.zh.md`](PROTOCOL.zh.md) (authoritative if the two disagree).

## 1. Goals and non-goals

**Goals**

- Any vendor, on their own machine, follows this protocol and produces a self-consistent, checkable result package containing "single-P capacity" and "single-D capacity".
- The evaluator puts several vendors' results into one table for cross-hardware comparison, and re-measures suspicious numbers.
- Onboarding requires only: **bring up two OpenAI-compatible endpoints + fill in one manifest + run our kit**. No machine access is given to the evaluator, and the server does not have to use the evaluator's image.

**Non-goals**

- Does not measure real end-to-end PD (handoff cost, queueing, KV transfer).
- Does not measure multi-node interconnect or cluster scalability.
- Does not replace the common gates (single-GPU compute, NCCL, …).

## 2. Layering and trust model

| Layer | Owner | Content |
|---|---|---|
| Endpoint | Vendor | Bring up a P and a D instance with vLLM (upstream, or the vendor's fork/plugin), and declare the configuration truthfully |
| Protocol constants | kit | SLO, scenario, concurrency tiers, request counts, temperature, seed, cold-start rules |
| Data | kit | Dataset generator, tokenizer check, fingerprints |
| Load generation and collection | kit | Pinned `vllm bench serve` + metric collection |
| Self-checks | kit | Blocking checks (§8); no result is produced if they fail |
| Evaluation and publication | Evaluator | Cross-vendor comparison, comparison against the reference platform, re-measurement, reporting (not in this repository) |

**Trust model: the vendor self-measures, we re-measure.**

- Vendor side: the kit automatically rejects honest misconfiguration (too few requests, hit rate off target, fingerprint mismatch, output length not taking effect).
- Evaluator side: re-measure with **a different seed set** held by the evaluator. If the re-measured value deviates from the self-reported one by more than ±5%, the vendor is first asked to explain; if the explanation does not hold, the re-measured value prevails (§9).
- The kit does not claim to defend against deliberate cheating. Optimizations aimed at a fixed input (warming, memorizing the prefix) are exposed by the separate seed set.

**Decisions already made**

1. The hardware under test includes non-NVIDIA → the protocol is defined at the OpenAI-compatible HTTP endpoint layer and assumes no CUDA.
2. The engine is pinned to vLLM (upstream, or a vLLM-based fork/plugin). Non-vLLM stacks are out of scope for this version.
3. The dataset is synthetic and publishable: the content is random placeholder tokens; **every request has the same input length** (one shared prefix + a fixed-length unique suffix); **the output length is fixed too** — 1 token per request for the P test, 1,200 tokens per request for the D test (§4). Generation reads no real data: the dataset contains no real content and no real request lengths.
4. Precision: try the official Hy3-FP8 weights first; another precision is allowed only if those cannot run, and it must be declared. **v1 has no precision gate** (§7): the result package is marked "accuracy unverified", and the accuracy-alignment suite comes later.

## 3. Interface under test (endpoint contract)

The vendor provides two endpoints, used **one at a time** (a single-role test loads only one role, so P and D need not be online simultaneously):

| Role | Required capability |
|---|---|
| P (prefill-only test) | `POST /v1/chat/completions`; supports `max_completion_tokens`; `ignore_eos` takes effect; prefix cache on |
| D (decode-only test) | Same, and **the prefix-cache hit rate must be measurable** |

**How the hit rate is measured (in priority order)**

1. The delta of `prefix_cache_hits_total` / `prefix_cache_queries_total` from the Prometheus `/metrics` endpoint (every vLLM-family server has these; this is what the reference platform used).
2. Summing `usage.prompt_tokens_details.cached_tokens` over responses.
3. Neither available → **the D result is invalid** (there is no way to prove that only decode was measured).

The vendor declares which one is used in the manifest; the kit verifies against that declaration and fails if the declaration and the measurement disagree.

**Start-up requirements**

- The machine is exclusive during the test; no other GPU load.
- Clear the compilation cache before each run, so every run starts from the same cold-compile state; the kit records whether it was cleared.
- The P and D instances **do not need** a KV-transfer connector. On the reference platform, removing `NixlConnector` left the P/D metrics unchanged (§14), so vendors need not install NIXL/UCX and need not configure a data-plane NIC.

**Manifest `vendor_manifest.yaml` (all fields required)**

```yaml
protocol_version: pdbench-1.0
vendor: <vendor>
run_id: <vendor-defined, unique>
hardware:
  gpu_model: <model>
  gpu_count: <N>            # GPUs per node
  gpu_mem_gib: <N>
  cpu_model: <model>
  cpu_cores: <N>
  ram_gib: <N>
  driver: <version>
  cuda: <version>
image:
  ref: <image reference>     # or the sha256 of a `docker save` archive
  digest: <sha256>
engine:
  vllm_version: <version>
  fork: <repo/commit, or "upstream">
  deviations: [<replaced kernels / custom attention / scheduler changes / unmerged patches …>]
precision:
  weights: <fp8 | bf16 | int8 | ...>
  kv_cache: <fp8_e4m3 | auto | ...>
  rationale: <if weights != fp8, why the official Hy3-FP8 cannot run>
endpoints:
  p: http://<host>:<port>
  d: http://<host>:<port>
  metrics: http://<host>:<port>/metrics
  hit_rate_source: metrics | usage   # see §3
serve:
  p_cmd: <full start command, including every environment variable>
  d_cmd: <full start command>
  tp: <N>
  max_model_len: <N>
  gpu_mem_util: <0.x>
  prefix_caching: true               # required for D
  chunked_prefill: <true/false/default>
  mtp_tokens: <0 | N>
  cudagraph: <description>
  other_optimizations: [<...>]
client:
  image: <provided by the kit; the vendor must not substitute it>
  cpuset: <reserved CPU cores, see §12-O2>
```

## 4. Fixed protocol constants (unchanged across hardware)

| Item | Value |
|---|---|
| `protocol_version` | `pdbench-1.0` |
| Scenario | 256k: **every request has the same input length**, by design 84,992 tokens = 78,192 shared prefix + 6,800 fixed-length unique suffix (designed hit rate 0.92). The actual token count is slightly below the design value because of boundary alignment and special tokens; C6 checks it within ±2% |
| Output | **Fixed length**: 1 token per request for the P test; **1,200** tokens per request for the D replay pass (`--custom-output-len 1200`), with `ignore_eos=true`. The dataset files still carry an `output_tokens` column (a historical field the generator wrote from a log-normal distribution); **the kit does not use it**, and it is kept only so that the dataset fingerprint does not change. Why variable-length output was dropped: §13 |
| SLO | mean TTFT ≤ 7,000 ms (P); mean TPOT ≤ 20 ms (D). p99 is reported alongside but does not drive the main verdict |
| Temperature | **explicit `temperature=0.9` (sent for both P and D)**; `seed=0` |
| Request rate | `request_rate=inf` (closed loop) |
| Concurrency tiers | kit defaults — P: 1–8; D: 1, 2, 4, 6, 8, 10, 12, 16, 20 (ceiling c=20, see "Concurrency-tier ceiling" in §6), together with `--stop-after-fail 2` to stop after two consecutive tiers miss the SLO. `first_fail_concurrency` depends on which tiers were measured; **the capacity verdict does not** |
| Request count | P: 80 per tier (the ≥10×c rule requires c≤8); D: `min(N, max(40, 10×c))` per tier, where `N` is the number of prompts actually filled (see §6 step 2) |
| D cache fill | Before replaying, run one pass over `N` prompts (`max_completion_tokens=1`, c=4), where `N = min(replay-set size, max(40, 10×highest tier))`; then replay tier by tier |
| warmup | 10 per tier for P; 2 per tier for D (excluded from the statistics) |
| Repeats | **≥2 independent runs per cell** (vendor self-measurement runs twice on the same instance; cross-machine repetition is listed as uncovered in §14), and report the spread |
| Throughput metrics | QPM = requests per minute (`request_throughput × 60`). TPM = tokens per minute, in three parts: `input_tpm` (input, including the tokens served from the prefix cache), `output_tpm` (output) and `total_tpm` (their sum); taken from the client's `output_throughput` and `total_token_throughput` times 60 — the same source as the `Output/Total token throughput (tok/s)` lines the client prints. Per-GPU TPM = `total_tpm ÷ serve.tp` |
| Capacity definition | **If every measured tier PASSes, `capacity.knee_found` is false and the result reads as "capacity >= the highest tier measured, knee not found"; it must not be presented as the hardware's capacity.** The primary reading is `max_tpm` = the PASS tier with the highest `total_tpm`; `max_qpm` = the PASS tier with the highest QPM is reported alongside. D additionally reports `contiguous` = the highest tier such that every tier from the smallest up to it PASSes (D's primary metric; selected by tier order, independent of which metric is maximised). All three entries carry QPM, steady-state QPM (`c×60000/mean_e2el_ms`), `output_tpm`, `total_tpm` and `per_gpu_total_tpm` |

**On temperature**: both P and D send `temperature=0.9` explicitly rather than relying on the server's `generation_config` default, because defaults are not guaranteed to agree across engines and images. Sending it explicitly was verified on the reference platform to leave the P metrics unchanged (§14).

**How to read TPM**:

- The P test emits 1 output token per request, so its `output_tpm` equals QPM numerically and carries nothing beyond QPM; read P throughput from `input_tpm` and `total_tpm`.
- `input_tpm` counts the tokens served from the prefix cache (this scenario's designed hit rate is 0.92), so it is not prefill compute throughput; the uncached part is `input_tpm × (1 − prefix_hit)`.
- With the D output fixed at 1,200, `output_tpm = QPM × 1,200` is an identity. Neither the SLO verdict (mean TPOT ≤ 20 ms) nor the selection of the contiguous-PASS tier changes with the choice of metric.
- `max_tpm` and `max_qpm` can land on different tiers. P input length is constant by design (C6 allows ±2%), so when the two differ the difference comes from input-length jitter across the per-tier datasets and does not represent a performance difference; compare the two tiers' `mean_input_tokens` and `mean_output_tokens` before reading anything into it.
- The denominator of per-GPU TPM is the manifest's declared `serve.tp` (the GPUs the instance actually uses), recorded in `capacity.per_gpu_basis`. Without a manifest `per_gpu_total_tpm` is `null` rather than divided by a guessed count. Reports must state whether the denominator is the TP degree or the whole machine's GPU count.
- The spread check (C9) still looks only at latency and QPM: with the D output fixed, TPM spread is identical to QPM spread, and for P it is a near-duplicate.

## 5. P test

1. Bring up the P endpoint and wait until it is ready; record `GPU KV cache size` and every start-up argument.
2. For each tier: sample the `/metrics` hit counters → run `vllm bench serve` (the dataset file for tier c, with `--custom-output-len 1`, `--num-prompts 80`, `--max-concurrency c`, `--num-warmups 10`, `--temperature 0.9`, `--seed 0`, `--ignore-eos`) → sample the hit counters again.
3. Record: mean/p50/p99 TTFT, QPM, steady-state QPM, `input_tpm`/`output_tpm`/`total_tpm`, hit rate (counter delta), mean input tokens, completed count.
4. Verdict: mean TTFT ≤ 7 s; capacity is `max_tpm` and `max_qpm` per the capacity definition in §4.

## 6. D test (two-pass method)

**Why two passes**: a single-role D test cannot depend on KV transfer from the P side, and a fake KV connector (`DecodeBenchConnector`) collapses the MTP acceptance rate to 2.5%, which makes it unusable. Instead, D fills its own prefix cache: the same batch of prompts is first sent with `max_completion_tokens=1` (prefill only, leaving the KV resident), and then replayed tier by tier — the replay hits close to 100%, so it is essentially decode only.

1. Bring up the D endpoint (prefix cache must be on).
2. Fill the cache: `N` prompts, c=4, `max_completion_tokens=1`; record this pass's hit rate. **`N` is derived from the tiers you intend to measure** — `N = min(replay-set size, max(40, 10×highest tier))`. Do not fill the replay set to the brim (see "Why N is not maximized" below).
3. **Per-tier re-touch**: before replaying a tier, send one pass over exactly the prompts that tier will use, with `max_completion_tokens=1` and c=4, and record that pass's hit rate as `tiers[].touch_hit`. Prompts still in the cache cost almost nothing; prompts that were evicted are recomputed and become most-recently-used again. **This pass is outside the measurement window and does not enter the tier's metrics.**
4. Replay tier by tier: same dataset, same seed, `--custom-output-len 1200` (1,200 output tokens per request), request count per the formula in §4.
5. Record per tier: mean/p99 TPOT, TTFT, QPM, steady-state QPM (with a fixed output length the two should nearly coincide; a large gap means something is wrong in the tail of the run or in server-side queueing), `input_tpm`/`output_tpm`/`total_tpm`, hit rate, `touch_hit`, MTP acceptance rate (if MTP is on), peak running/waiting, peak KV utilization.
6. Verdict: mean TPOT ≤ 20 ms.

**Why step 3 is needed**: if the cache is filled only once at the start, the prompts that no tier has used yet are never touched again after the fill pass, which makes them the oldest entries in LRU order; meanwhile the 1,200 output tokens of every request also occupy KV, and that accumulates as the tiers progress. Once the total approaches the KV capacity, those prompts are evicted first. The symptom is that **the tier's first run shows a lower hit rate and a longer time, and the second run recovers**, so the two repeats diverge. Measured on the reference platform: two configurations started showing this from c=10 and c=14 respectively, the starting tier was identical on two machines, and the two repeats differed by roughly 5%–9% (that tier's TPOT and QPM). `touch_hit < 1` means eviction had happened before that tier started. **Hardware with less memory triggers this at a lower tier.**

**Prerequisite (capacity rule)**: the resident footprint of the fill pass ≈ `78,192 shared prefix + N × 6,800` (the shared prefix is stored once; `N` is the number of prompts actually filled, see step 2). The vendor declares `GPU KV cache size` from the start-up log, and the kit checks:

```
capacity ≥ (78,192 + N×6,800 + c_max×6,800) × 1.05
```

**This formula counts only the prompt footprint; it does not include the accumulated 1,200 output tokens per request** — that part grows with the tiers and is absorbed by the per-tier re-touch in step 3, which does not change the formula. Worked values: highest tier 20 → N=200 → requires ≥ 1,652,902 tokens; highest tier 8 → N=80 → requires ≥ 710,422 tokens. If the capacity is insufficient, solve the formula for a smaller `N`, record it in the result package (`fill.degraded=true`); if `N` falls below a tier's request count, that tier's D result is marked "limited" and excluded from the capacity verdict. **This rule is derived analytically. On the reference platform the fill pass's hit rate ≈ the shared-prefix fraction and the replay pass ≈ 100%, consistent with the derivation, but it has not been measured on a small-memory card.**

**Concurrency-tier ceiling**: a tier's request count is capped by the replay-set size, `min(fill_n_max, replay-set lines)`, while C4 requires at least 10×concurrency requests per tier. The two cannot both hold once `10×c > available lines`, so such a tier is bound to fail C4. With the current replay set (200 prompts) the **D concurrency ceiling is therefore c=20**; the kit rejects over-ceiling tiers before starting (exit code 2), and the default D tiers are `1 2 4 6 8 10 12 16 20`. **If the hardware's own D capacity is ≥ 20, the current replay set cannot find its knee** — that requires a larger replay set and a larger `fill_n_max` first, which changes the dataset fingerprint and is therefore a protocol-level change.

**Why N is not maximized**: KV holds two kinds of data — the resident replay set (`shared prefix + N×suffix`) and the **output blocks of past requests** (1,200 tokens each, accumulating across tiers, not counted by the capacity formula). Once their sum crosses the KV capacity the server evicts by LRU, and the first victims are the replay-set prompts this sweep has not reached yet; the re-touch for that tier then has to do a real prefill (which costs time only and does not affect the result — see step 3). The larger `N` is, the larger the resident share, and the lower the tier at which eviction starts. Measured on the reference platform (highest tier 20): with N=200 the resident replay set occupies about 67% of KV and eviction starts from c=14; estimated at the same capacity, N=250 occupies 83% (eviction from c=12) and N=300 occupies 99% (from c=2). **The accumulated total at the end of a sweep necessarily exceeds KV (measured at roughly 2.2×), so eviction cannot be eliminated by shrinking N** — the choice of N only decides how many tiers need a real prefill. So take `N` just large enough for the tiers you want: `c_max = N/10`, and measuring beyond the knee serves no purpose. When the knee is unknown, bracket it with a coarse sweep at a small `N` first, then set `N` for the fine sweep.

## 7. Precision rules

1. **Try the official Hy3-FP8 weights first.** If the vendor's hardware and kernels can run them at all, they must be used; this is not a trade-off.
2. If they cannot run, another precision is allowed, but the vendor must:
   - declare the weights / KV cache / activation quantization scheme and the reason in the manifest;
   - carry a precision label in the result table, and **not be compared in the same column as FP8 results**.
3. Capacity numbers at a self-chosen precision are for reference only and must not be presented as that hardware's "FP8-equivalent capacity".
4. **v1 has no precision gate**: `accuracy.status` is fixed to `unverified` in every result package, and the report states plainly that the capacity numbers have passed no quality gate. The reason: the load test uses random placeholder tokens, so the output is meaningless and **the performance data itself cannot tell whether the output is correct**. This is a known gap, not a risk that has been ruled out (the fp8 KV precision of the reference platform itself is also unvalidated).
5. **Smoke tier (optional, provided by the kit)**: about 50 prompts, greedy, checking only that the output is not garbled, not repetitive, and terminates normally; written to `accuracy.smoke`. It cannot catch subtle precision loss.
6. Upgrade trigger: when the first vendor onboards, or when the numbers are to be used for purchasing or selection decisions, upgrade to the "standard tier" (run a subset of a public benchmark suite with off-the-shelf tooling and compare against the reference platform's FP8 scores).

## 8. Automatic self-checks in the kit (blocking)

C2t, C13 and C14 are **pre-flight checks**: they run before `run` starts and exit immediately on failure (`--continue-on-preflight-fail` is for debugging only). The remaining checks are evaluated after the run. Any FAIL → `run` exits with code 1; `result.json` is still written but with `submission_grade=false` (to help diagnosis). The exit code looks only at FAIL. `submission_grade=true` additionally requires a manifest and that **every check is PASS or WARN** (any SKIP, for example because `--server-log` was not given, makes it false).

| # | Check | Threshold |
|---|---|---|
| C1 | The manifest's `protocol_version` matches the kit's; the kit's own sha256 is recorded in the result. SLO/scenario/temperature/seed are **not** cross-checked (they are built-in kit constants, not vendor-supplied) | No manifest → SKIP |
| C2 | The combined fingerprint of the 9 base dataset files = `e9beb87e24e1` (with `--repeat>1`, the repeat-set fingerprint `9adcdac59475` is checked as well). **Only the combined fingerprint is checked**; files are not reported individually | Exact match |
| C2t | The combined fingerprint of the three tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) = `7f062efabe6a` (byte-identical to the three files published by Hugging Face `tencent/Hy3-FP8`). `chat_template` determines the prompt the server actually sees | Exact match |
| C13 | Client self-check: drive a local stub with the real client command line; the set of request-body hashes received must equal the golden set (`kit/selftest/golden_requests.json`); and the client's `vllm --version` must equal the pinned version `0.24.0` | Exact match |
| C14 | Server probe: send a small streaming request shaped exactly like the real client's (`content` as a segment list); the stream must carry a complete `usage`, and asking for 8 output tokens must yield 8 (`ignore_eos` in effect). The client takes the input token count from the server's `usage.prompt_tokens` and silently falls back to its own count when that is missing, which would void C6 — hence this separate check | Pass |
| C15 | The server-side preemption counter delta (`vllm:num_preemptions_total`) must be 0 for every run — when KV is tight the server preempts running requests and recomputes them, which inflates that tier's latency while the hit rate shows nothing | Metric unavailable → SKIP |
| C3 | Every run completes the planned number of requests with no errors | 100% |
| C4 | Each tier has at least 10×concurrency requests | — |
| C5 | Actual hit rate (from `/metrics` counter deltas): P must be within [0.90, 0.95] on every run (the upper bound catches "the suffix got cached"); D replay must be ≥ 0.999 on every run (the lower bound catches "the replay set was evicted"; raised from 0.98, because the old bound let through runs that had been evicted and re-prefilled) | As stated |
| C6 | Mean input tokens per run = 84,992 ±2% | As stated |
| C7 | P: per-request output length is 1 for every request. D: per-request output length is 1,200 for every request (`ignore_eos` in effect, nothing truncated; not being able to read per-request lengths is also a FAIL). C7f: the fill pass completed in full | Exact match |
| C8 | The client's warmup count matches the protocol (verified from the client log). "Compilation cache cleared" and "machine exclusive" are **declarations in the manifest only**; the kit cannot verify them | No manifest → FAIL |
| C9 | At least 2 runs per tier; relative spread of TTFT/TPOT and QPM across repeats ≤1% passes, 1%–3% WARN, >3% FAIL; a single run → FAIL (SKIP when there is no manifest) | Thresholds taken from the measured same-instance spread on the reference platform |
| C10 | The declared `hit_rate_source=metrics` and `/metrics` is actually collectable. The `usage` source is unimplemented in the kit → SKIP | No manifest → SKIP |
| C11 | `--server-log` was given and the log contains `GPU KV cache size`. **Only the presence of that line is checked**; the start command is not cross-checked against the manifest | No log → SKIP |
| C12 | D: KV capacity ≥ (78,192 + N×6,800 + c_max×6,800) × 1.05 (§6) | No KV data → SKIP |

## 9. Re-measurement and adjudication

- When re-measurement happens: spot checks across all vendor results (at least one per batch is recommended, prioritizing the "unusually good" ones); or the C9 spread exceeds its limit; or the vendor chose their own precision (precision being unvalidated, re-measurement is the only external check).
- How: same machine, same image, same start command, **with the evaluator's private seed set** (same distribution, different `offset-base`); the protocol itself is unchanged.
- What is compared: two numbers — **single-P capacity QPM** and **single-D capacity QPM (contiguous)**. Deviation = |re-measured − self-reported| / self-reported.
- Deviation ≤ 5%: accept the self-reported value and note in the report that it was re-measured.
- Deviation > 5%: ask the vendor to explain. Deviations with a valid explanation (temperature/clock throttling/noisy neighbours) are recorded; otherwise the re-measured value prevails and the report says so.
- Re-measured values are always archived separately and never overwrite the vendor's self-reported values.

## 10. Scope of this repository

This repository contains: the protocol (this document), the kit (`kit/`), the client image definition (`client/`), and the manifest template plus a reference adaptation (`templates/`, `examples/`).

**Not included**: the evaluator's cross-vendor evaluation, the re-measurement workflow, the evaluator's separate seed set, and any vendor's results. A vendor needs nothing beyond this repository to self-measure and produce a result package.

## 11. Directory layout and usage

```
pd_single/                           # this directory: the "single-node P/D test" sub-module of inference_bench/
├── PROTOCOL.md                      # this document (English)
├── PROTOCOL.zh.md                   # Chinese version (authoritative if the two disagree)
├── kit/
│   ├── pdbench.py                   # gen-datasets / validate-manifest / selftest-client / run (P and D tests + self-checks + result package)
│   ├── dataset/                     # gen_prefix_dataset.py, gen_dataset.py
│   ├── selftest/                    # client self-check: prompts.jsonl (3 fixed prompts), golden_requests.json (request-body hashes)
│   └── tests/test_checks.py         # regression cases for the self-checks
├── client/Dockerfile                # pinned client image (validation scope: §14)
├── templates/vendor_manifest.yaml   # vendor manifest template (unfilled fields are rejected by validate-manifest)
└── examples/h20_vllm/               # reference adaptation: h20_manifest.yaml, serve_p.sh, serve_d.sh
```

**Not built yet**: `schema/result.schema.json` (the skeleton below is illustrative only). The client is pinned to the public release `vllm==0.24.0`; the validation scope of `client/Dockerfile` is in §14.

```bash
python3 -I kit/pdbench.py gen-datasets --tokenizer <model dir> --out-dir <dataset dir>   # generate and verify fingerprints (including the repeat sets)
python3 -I kit/pdbench.py validate-manifest vendor_manifest.yaml
python3 -I kit/pdbench.py selftest-client --model-path <model dir>    # run this before testing: checks the client request body (no server needed)
python3 -I kit/pdbench.py run --role p --base-url http://<host>:<port> --model-path <model dir> \
    --served-model-name <name> --dataset-dir <dataset dir> --out <output dir> --manifest vendor_manifest.yaml --server-log <P server log>
python3 -I kit/pdbench.py run --role d ... --stop-after-fail 2        # --repeat defaults to 2; omitting --tiers uses the default tiers
```

Running the kit needs Python ≥ 3.8 (older versions exit with a message pointing at the client image). Build and run notes for the client image are in the header comment of `client/Dockerfile`.

When a self-check fails the process exits with code 1 and `result.json` carries `submission_grade=false`. `run` also writes `summary.csv` (one row per concurrency tier, including the mean across repeats).

**Result package**: `result.json` (primary) + `summary.csv` + the raw `c<N>_r<k>.json/.log` for each tier (including per-request `ttfts/itls/output_lens`; `generated_texts` is stripped by default). The cross-vendor evaluation reads `result.json` and is not part of this repository.

### result.json skeleton

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
      "qpm": 11.1, "steady_qpm": 11.1,
      "input_tpm": 888888.8, "output_tpm": 11.1, "total_tpm": 888899.9, "prefix_hit": 0.93,
      "verdict": "PASS", "repeat": [{"qpm": 11.1, "mean_ttft_ms": 1111.1, "p99_ttft_ms": 2222.2}, {"...": "..."}]
    }],
    "capacity": {
      "max_tpm": { "concurrency": 4, "qpm": 11.1, "steady_qpm": 11.1, "output_tpm": 11.1,
                   "total_tpm": 888899.9, "per_gpu_total_tpm": 111112.5 },
      "max_qpm": { "concurrency": 4, "qpm": 11.1, "steady_qpm": 11.1, "output_tpm": 11.1,
                   "total_tpm": 888899.9, "per_gpu_total_tpm": 111112.5 },
      "first_fail_concurrency": 5, "knee_found": true, "max_tier_measured": 8,
      "per_gpu_basis": { "source": "serve.tp", "value": 8 }
    }
  },
  "d_only": {
    "fill": { "num_prompts": 200, "completed": 200, "hit_ratio": 0.92 },
    "tiers": [{
      "concurrency": 8, "num_prompts": 80, "completed": 80,
      "mean_input_tokens": 84800, "mean_output_tokens": 1200,
      "mean_ttft_ms": 333.3, "mean_tpot_ms": 9.99, "p99_tpot_ms": 9.99,
      "qpm": 22.2, "steady_qpm": 22.2,
      "input_tpm": 888888.8, "output_tpm": 26640.0, "total_tpm": 915528.8,
      "prefix_hit": 1.0, "touch_hit": 1.0, "preemptions": 0,
      "mtp_accept": null, "verdict": "PASS", "repeat": [{"...": "..."}]
    }],
    "capacity": {
      "contiguous": { "concurrency": 8, "qpm": 22.2, "steady_qpm": 22.2, "output_tpm": 26640.0,
                      "total_tpm": 915528.8, "per_gpu_total_tpm": 114441.1 },
      "max_tpm": { "concurrency": 8, "qpm": 22.2, "steady_qpm": 22.2, "output_tpm": 26640.0,
                   "total_tpm": 915528.8, "per_gpu_total_tpm": 114441.1 },
      "max_qpm": { "concurrency": 8, "qpm": 22.2, "steady_qpm": 22.2, "output_tpm": 26640.0,
                   "total_tpm": 915528.8, "per_gpu_total_tpm": 114441.1 },
      "first_fail_concurrency": 10, "knee_found": true, "max_tier_measured": 12,
      "per_gpu_basis": { "source": "serve.tp", "value": 8 }
    }
  },
  "accuracy": { "status": "unverified", "smoke": null },
  "checks": [ { "id": "C2", "status": "PASS", "detail": "dataset fingerprint e9beb87e24e1" } ]
}
```

> The numbers above are **made-up placeholders** (11.1, 22.2 and the like) and do not represent any hardware's measurements. In practice `run` writes one `result.json` per role, and the kit's output defines the fields.

## 12. Open / unvalidated items

| # | Item | Notes and proposal |
|---|---|---|
| O1 | Accuracy-alignment suite (**later**) | v1 has no gate (§7). Later: smoke tier (half a day) → standard tier (an lm-eval subset compared against the reference platform's FP8 scores, 1–2 days) → full tier (including long context, more than a week; the hard part is setting thresholds). |
| O2 | Client resource isolation unvalidated | Today the client runs on the same machine and in the same container as the server; the effect of CPU contention has not been measured. Proposal: reserve a fixed number of vCPUs and declare them in the manifest's `client.cpuset`; start with a "same machine vs isolated" comparison. |
| O4 | Fill-size down-scaling | The rule is defined (the capacity formula in §6); the kit only checks it and **does not down-scale automatically**; untested on a small-memory card. |
| O4b | Concurrency-tier ceiling | A 200-prompt replay set ⇒ a D tier ceiling of c=20 ("Concurrency-tier ceiling" in §6). The kit rejects over-ceiling tiers before starting but **does not enlarge the replay set**; hardware with D capacity ≥ 20 needs a protocol-level change first (a larger `fill_n_max` and replay set, and a new fingerprint). |
| O6 | Handling missing capabilities | A stack that does not support `max_completion_tokens` / `ignore_eos` / prefix cache → judged "not applicable" with an explanation, and no capacity number is produced. |
| ~~O7~~ | ~~Dataset distribution~~ | **Decided**: ship the generator plus the fingerprints; the vendor generates locally and the kit verifies (the packaged dataset is about 1 GB and is not distributed with the repository). Hy3 is an Apache-2.0 open model (Hugging Face `tencent/Hy3-FP8`), so its tokenizer may be redistributed with the kit; the kit nevertheless does **not** bundle it — the vendor has to download the full model directory to serve it anyway, and `gen-datasets --tokenizer` and C2t use that directory directly, which avoids keeping a second, possibly stale copy in the repository. |
| O8 | How add-on configurations enter the tables | The main table carries only the main configuration (no MTP, no speculative decoding); MTP and other accelerations are listed separately as "add-on configurations" and are not mixed into the main comparison. |
| O12 | Implementation gaps | The `usage.cached_tokens` hit-rate source, automatic `N` down-scaling, result-package bundling, a formal JSON Schema for `result.json`, cross-machine repetition. |
| O14 | Server-side compatibility with the request body | The client sends `content` as a segment list (see the request body below); how different engines and chat templates handle that is unknown. The C14 probe exposes outright rejection but **cannot expose "accepted, but a different prompt was cut out"** — only C6 (the server's `prompt_tokens`) can reveal that indirectly. |

**Client request body** (confirmed byte-identical between vLLM `0.24.0` and the earlier development build by capturing against a stub):

```json
{"model": "<served-model-name>", "messages": [{"role": "user", "content": [{"type": "text", "text": "<prompt>"}]}],
 "max_completion_tokens": <N>, "stream": true, "stream_options": {"include_usage": true}, "ignore_eos": true, "temperature": 0.9}
```

Note that `content` is a **segment list**, not a string. The client takes the input token count from the server stream's `usage.prompt_tokens`; when the server omits `usage` the client silently falls back to its own count, which voids C6 — hence C14.

## 13. Approaches already ruled out (do not retry)

- The `DecodeBenchConnector` fake KV: it collapses the MTP acceptance rate to an unusable level.
- Too few requests per tier (for example a fixed 60): this clearly underestimates QPM, because concurrency is not saturated in the tail of the run; changed to ≥10×concurrency.
- Reporting only the mean and not p99: the two can imply noticeably different capacities, so both must be reported.
- Transcribing start commands into the report by hand: they drift from the scripts; the commands in the manifest should be generated programmatically from the configuration.
- Bringing up a KV-transfer connector for P/D in a single-role test (§14: no effect on the reference platform).
- Variable-length output for the D test (reading each row's `output_tokens`, log-normal with mean 1,200): ① each tier takes only the first N rows of the replay set, so the actual mean output length differs per tier (it moved within roughly ±3% on the reference platform), QPM drifts with it, and adjacent tiers are not comparable; ② near the end of a run the remaining long requests run alone at low concurrency, stretching the total time, so the **steady-state QPM is 7%–13% higher than the measured QPM** (that is, the measured value is underestimated); ③ mean TPOT averages per request, giving short and long requests equal weight. After switching to a fixed 1,200, the comparison on the reference platform (c=4/6/8, two machines, two repeats each) showed: mean TPOT within 1%, the SLO verdicts and the capacity tier unchanged, p99 TPOT slightly lower, and the gap between steady-state and measured QPM down from 7%–13% to within 0.2%.
- Reusing the same dataset for repeated P runs: the unique suffix is already in the prefix cache on the second pass, so TTFT drops to a fraction of its value and prefill is effectively not measured (the kit uses separate `_r2/_r3` datasets and puts an upper bound on the hit rate).

## 14. Validation status

**Validated on the reference platform** (single node, 8 GPUs, Hy3-FP8, upstream-family vLLM; the actual numbers are not published):

- The kit's full-sweep P (8 tiers) and D (multiple tiers) runs agree tier by tier with the evaluator's own baseline; the `submission_grade` path, the pre-flight checks and the self-check regression cases (`kit/tests/`) pass.
- After switching D to a fixed output of 1,200, a comparison was run on the reference platform (§13): TPOT and the SLO verdicts unchanged, and the measured QPM no longer dragged down by the tail of the run. Reference values were then re-measured under the fixed-output rule for **all tiers of three D configurations plus three P configurations** — 9 results, all `submission_grade=true`, with each D configuration run once on each of two machines.
- The primary capacity reading changed from QPM to `total_tpm` (§4). TPM is taken from the client's own `output_throughput` and `total_token_throughput`, which adds no collection path. Checked against every retained per-run raw result on the reference platform: `output_tpm` is identical to `QPM × mean output length` within float precision, and `total_tpm` differs from the value derived from the aggregate rows by under 1e-5 relative (caused by the rounding of mean input tokens in those rows).
- The same 9 results (all tiers of three D configurations plus three P configurations — 81 tiers) were re-measured on the reference platform under the TPM reading: all `submission_grade=true`, with the C15 preemption counter 0 throughout. **In all 9, `max_tpm` and `max_qpm` land on the same tier** — this scenario's token lengths are constant per tier (fixed D output, dataset-fixed P input), so the change of reading alters the unit of the capacity result rather than which tier is selected. Compared tier by tier against the previous round: latency and QPM both differ by under 2% at most (inside C9's WARN band), with the capacity tier identical in every result; the only verdict difference falls on a tier whose mean TPOT sits under 0.3% away from the 20 ms SLO, and that configuration's `contiguous` reading is unaffected. **For a tier that sits on the SLO line, the "highest within SLO" reading is not guaranteed to reproduce at tier granularity.**
- The request body emitted by the pinned client `vllm==0.24.0` is byte-equivalent to that of the earlier development build (identical stub-captured hashes), and the two reconcile on every metric.
- Removing the KV-transfer connector, and sending `temperature=0.9` explicitly for P, leave the P/D metrics unchanged.
- `client/Dockerfile` built successfully on two machines, and `selftest-client` passes inside the image.

**Unvalidated / uncovered** (please be aware before use):

- **The D concurrency ceiling is c=20 and the P ceiling is c=8** (a consequence of the dataset size; see "Concurrency-tier ceiling" in §6 and O4b): hardware whose capacity exceeds those tiers cannot have its knee measured, and the kit rejects over-ceiling tiers before starting.
- Only upstream-family vLLM servers have been validated: **no non-NVIDIA hardware, no non-vLLM server, no small-memory card**.
- Repeats are two runs on the same instance; cross-machine repetition comes from "the same configuration run once on each of two machines", but the two machines do not run the same tier at the same time, so **machine-to-machine differences cannot be separated from drift over time**; C9 only catches same-instance random variation.
- The `usage.cached_tokens` hit-rate source and automatic `N` down-scaling are unimplemented.
- The default base image of `client/Dockerfile`, `python:3.12-slim`, is unvalidated (an equivalent Python 3.12 image was used instead); old Docker (19.03 as measured) needs `--security-opt seccomp=unconfined`, see the comments in the Dockerfile.
- Request-body equivalence only proves that **the request is the same**; it does not prove that the vendor's server treats it the same way (O14).
- **No accuracy check was performed** (§7), so none of the capacity numbers has passed a quality gate.
