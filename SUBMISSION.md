# Submission

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

## 6. Sign-off

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
