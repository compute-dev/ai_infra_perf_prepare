# Submission

**English** | [中文](SUBMISSION.zh.md)

What to hand back, and the tables to complete alongside it.

The machine-generated part is produced by `tools/collect_results.sh` and
`tools/summarize.py`. The tables below are the human-authored part — they capture
things a script cannot decide, such as what was intentionally excluded and why.

---

## 1. Package

```bash
bash tools/collect_results.sh --vendor <name> --gpu <model> --dir results
python3 tools/summarize.py results --vendor <name> --gpu <model>
```

Send the resulting `results_<vendor>_<gpu>_<date>.tar.gz`, which contains:

```
results/
  SUMMARY.md          auto-generated verdict per gate
  results.json        same content, machine-readable
  MANIFEST.txt        what is present / missing
  environment.txt     hardware + software snapshot of the collecting host
  preflight/          runtime_*.csv, spec_*.csv
  cublas/             cublas_bench_*.log and/or percard_cluster/<node>/dev<N>.log
  nccl/               nccltest_*.log, 1node_nccltest_*.log
  perftest/           perftest_*.log, summary_*.txt, summary_agg_*.txt
```

**A gate with no output has not been tested.** If a gate is missing, either run
it or say so explicitly in the notes below. `collect_results.sh` refuses to
package an incomplete set unless you pass `--allow-partial`, and the manifest
records what was missing either way.

### Coverage

| | |
| --- | --- |
| Nodes in scope | |
| Nodes actually tested | |
| Nodes excluded, and why | |
| GPUs per node | |
| GPU model | |
| NCCL scales tested (nodes) | |
| Inference gate run? (inference cards only) | |
| SSH port / user used | |

> Excluded nodes must be named here. An unstated exclusion is indistinguishable
> from a node that was never tested.

---

## 2. Environment acceptance

Fill in the actual value observed. Gate 0's CSVs are the evidence.

| Item | Required | Actual | Evidence (CSV field) |
| --- | --- | --- | --- |
| OS | Ubuntu 24.04 LTS or equivalent | | `os` |
| NVIDIA driver | ≥ 570 | | `driver` |
| CUDA | ≥ 12.9 | | `cuda_api` |
| User-space libcuda | consistent across nodes, not shadowed by cuda-compat | | `libcuda`, `cuda_compat` |
| Docker | ≥ 26 | | `environment.txt` |
| ECC mode | consistent across nodes | | `ecc_mode` |
| SM clock max | consistent across nodes | | `sm_clock_max` |
| Power limit | consistent across nodes | | `power_limit` |
| Compute HCAs | expected count, all ACTIVE | | `compute_hca`, `compute_active` |
| Fabric type | IB or RoCE | | `fabric_type` |
| RoCE: GID index, MTU, trust, PFC | identical across nodes | | `roce_gid_v2idx`, `compute_mtu`, `roce_trust`, `roce_pfc_prio` |
| OFED / fabricmanager / peermem | present and consistent | | `ofed`, `fabricmanager`, `peermem` |
| Kernel | identical across nodes | | `kernel` |
| GPU-NIC rail topology | identical across nodes | | `gpu_nic_rail` |
| PCIe ACS | disabled on every node | | see notes |
| Persistence mode | enabled | | `persistence` |
| memlock | unlimited (required for gate 3) | | `memlock` |
| Shared storage | *optional* — note path if used | | |

**Gate 0 result:** exit code `___` (0 = pass). Critical-field differences: `___`

---

## 3. cuBLAS compute acceptance

**Single-node quick check (GPU 0 only — not acceptance):**

| | |
| --- | --- |
| Image used | |
| `GPU_NAME` | |
| Result | pass / fail |

**Fleet per-card sweep:** `results/cublas/percard_cluster/`, aggregated into
`fleet_percard.csv`.

| Precision | Fleet min | Median | Max | Spread % | Baseline | Cards below baseline ×95 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP4 | | | | | | |
| FP8 | | | | | | |
| INT8 | | | | | | |
| FP16 | | | | | | |
| BF16 | | | | | | |
| TF32 | | | | | | |
| FP32 | | | | | | |

**Outlier cards** (below fleet median by >3 % on one or more precisions):

| Node | Card | Precision(s) | Deviation | Assessment |
| --- | --- | --- | --- | --- |
| | | | | |

Leave empty if none. For each outlier, state whether it is a **genuine weak card**
(consistently low across several precisions) or a **measurement artifact** — and
say what evidence supports that. Note in particular that an H800 TF32 reading
that is about half speed is a known cuBLAS artifact; judge TF32 on torch
`allow_tf32` instead.

**Gate 1 result:** pass / fail — `___`

---

## 4. NCCL interconnect acceptance

Values are **peak out-of-place busbw (GB/s) at the 16G message size**, the same
measurement `nccl_bench/baselines.md` uses. One table per scale tested.

| GPU × GPUs/node × nodes | AllReduce | All2All | AllGather | ReduceScatter | Baseline row matched |
| --- | ---: | ---: | ---: | ---: | --- |
| ×8×1 | | | | | |
| ×8×2 | | | | | |
| ×8×4 | | | | | |
| ×8×8 | | | | | |

**Out of bounds values** must be 0 in every log. Any non-zero value is a
correctness failure — record it and stop:

| Log | `# Out of bounds values` |
| --- | --- |
| | |

**Notes on judgement** (state these or the numbers may be misread):

- Cross-node All2All falling to ~10–20 % of the single-node figure is expected
  behaviour, not a defect.
- H800 AllReduce ≈215 GB/s is the reduced-NVLink hardware spec.
- If nodes were compared only within the same rail/leaf, say so — and say what
  `rail_isolation_check.sh` concluded.

**Gate 2 result:** pass / fail — `___`

---

## 5. RDMA point-to-point acceptance

**Rail isolation** (`rail_isolation_check.sh`):

| | |
| --- | --- |
| Same-rail bandwidth | Gb/s |
| Cross-rail | works / fails to connect |
| Verdict | rail-isolated / not rail-isolated |

**Per-rail single QP** (`summary_*.txt`) — expected ~193 Gb/s per 200G port:

| Rail | write Gb/s | bidir Gb/s | read Gb/s | write lat µs | send lat µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| | | | | | |

**Per-rail aggregate** (`summary_agg_*.txt`) — judge on **peak** over ≥3 runs:

| Rail | peak Gb/s | expected peak | runs producing a number |
| --- | ---: | ---: | --- |
| | | | |

**Error counters** — all five must be zero on every rail, before and after:

| Rail | out_of_sequence | packet_seq_err | req_cqe_error | local_ack_timeout_err | rnr_nak_retry_err |
| --- | ---: | ---: | ---: | ---: | ---: |
| | | | | | |

Any counter reported as `NA` means that rail is **unverified**, not clean — say so.

**Gate 3 result:** pass / fail — `___`

---

## 6. Inference acceptance (inference cards only)

Applies only to inference cards, and only after gates 0–3 have been accepted: a weak card
or a bad link contaminates these numbers. One submission covers **one hardware model and
one engine configuration**.

> **Status: draft, and outside the packaging scripts.** `tools/collect_results.sh` and
> `tools/summarize.py` do not yet know about this gate — package `results/inference/`
> yourself, by hand, alongside the tarball they produce. The protocol itself is a draft
> ([`inference_bench/pd_single/PROTOCOL.md`](inference_bench/pd_single/PROTOCOL.md)), and
> its open items are listed in that document's §12.

What is measured, per [`inference_bench/pd_single/`](inference_bench/pd_single/): single-node
**Prefill-only** and **Decode-only** capacity under a fixed SLO (mean TTFT ≤ 7 s for P, mean
TPOT ≤ 20 ms for D), 256k input at a 92% prefix-cache hit rate, model Hy3-FP8. End-to-end PD,
KV transfer and multi-node are **not** in scope.

| # | Item | Required value / evidence |
| --- | --- | --- |
| 1 | `submission_grade` | `true` in both `result.json` files. `false` means a self-check failed — fix it and re-run rather than submitting |
| 2 | Manifest | `vendor_manifest.yaml` with every field filled; `validate-manifest` exits 0 |
| 3 | Dataset fingerprints | `e9beb87e24e1`; with `--repeat>1` also `9adcdac59475`. Generated locally by `gen-datasets`, verified by the kit (C2) |
| 4 | Tokenizer fingerprint | `7f062efabe6a` — the three files from Hugging Face `tencent/Hy3-FP8` (C2t) |
| 5 | Client | The pinned image, `vllm==0.24.0`; request-body hashes equal the golden set (C13). Do not substitute your own client |
| 6 | Repeats | ≥2 independent runs per concurrency tier; report the spread. A spread above 3% blocks the submission (C9) |
| 7 | Single-P capacity | Highest QPM among PASS tiers, reported as both measured and steady-state, with the concurrency tier |
| 8 | Single-D capacity | QPM of the highest tier such that every tier below it also PASSes, reported as both measured and steady-state, with the concurrency tier |
| 9 | Knee evidence | The last PASS tier and the first FAIL tier, both with numbers. When `capacity.knee_found` is false (no tier broke the SLO), report it as "capacity ≥ `capacity.max_tier_measured`, knee not found" — **not** as the capacity |
| 10 | Accuracy | `accuracy.status` is `unverified` in this version. State plainly that the capacity numbers passed no quality gate |
| 11 | Server logs | The start-up log of each endpoint, containing `GPU KV cache size` (C11) |
| 12 | Preemption | Per-tier `preemptions` must be 0 (C15). A tier that was preempted has unusable latency numbers |

**Tier ceilings.** With the shipped dataset the D concurrency ceiling is c=20 and the P ceiling
is c=8 — a property of the dataset size, not of your hardware. The kit rejects over-ceiling
tiers before starting. If your capacity exceeds them, say so in the sign-off: the knee cannot
be measured without a protocol-level change.

### Result layout

```
results/
  inference/
    pd_single/
      p_only/
        result.json          primary result for the P role
        summary.csv          one row per concurrency tier
        c<N>_r<k>.json/.log  raw data per tier per repeat
      d_only/
        result.json          primary result for the D role
        summary.csv
        c<N>_r<k>.json/.log
      vendor_manifest.yaml   the manifest used for this run
      serve_p.log            P endpoint start-up log
      serve_d.log            D endpoint start-up log
```

Keep the two roles in separate directories: one `run` writes one role. A role with no output
has not been tested — say so in the sign-off rather than omitting it silently.

### Capacity table

| | Concurrency tier | Measured QPM | Steady-state QPM | Knee found | Notes |
| --- | --- | ---: | ---: | --- | --- |
| Single-P capacity | | | | | |
| Single-D capacity | | | | | |

**Inference gate result:** pass / fail — `___`

---

## 7. Sign-off

| | |
| --- | --- |
| Vendor | |
| Cluster / site | |
| Date | |
| Overall verdict (from `SUMMARY.md`) | |
| Known gaps or caveats | |
| Contact | |

> Report honestly. An inconclusive measurement that is labelled as such is
> recoverable; a number that is wrong and gets built on is not. If a gate could
> not be run, say which and why rather than omitting it.
