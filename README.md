# AI infrastructure performance — vendor preparation & acceptance package

**English** | [中文](README.zh.md)

This repository is the **vendor-facing** package for a GPU cluster POC. It contains
everything needed to prepare a cluster and produce the acceptance measurements,
runnable **by the vendor on the vendor's own cluster**: the scripts take your host
list, your SSH port and your output directory, and depend on nothing internal to us.

Run the four gates in order. Each one produces hardware-independent evidence that
can be reviewed, re-run and compared.

## The four gates

| # | Gate | Directory | What it proves | Rough time |
| --- | --- | --- | --- | --- |
| 0 | **Node consistency** | [`preflight/`](preflight/readme.md) | The nodes are actually identical. `driver_version` matching is **not** enough — drift here silently invalidates every later comparison. | minutes |
| 1 | **Single-GPU compute** | [`cublas_bench/`](cublas_bench/readme.md) | Per-card GEMM throughput (FP8/INT8/FP16/BF16/TF32/FP32, FP4 where applicable), across **every** card, with outlier detection to find weak or throttled boards. | ~10 min/cluster |
| 2 | **NCCL interconnect** | [`nccl_bench/`](nccl_bench/readme.md) | Intra-node NVLink and inter-node RDMA bandwidth for AllReduce / All2All / AllGather / ReduceScatter, at several cluster sizes. | ~30 min |
| 3 | **RDMA point-to-point** | [`perftest/`](perftest/readme.md) | The raw fabric underneath NCCL, per rail. Separates a fabric fault from an NCCL/config fault and catches packet errors that bandwidth alone hides. | ~20 min |

Gate 0 must come first: without consistent nodes the later numbers are not
comparable. Gates 1 and 2 are the headline acceptance results. Gate 3 is the
diagnostic that tells you *where* a gate-2 problem lives.

## Quickstart

```bash
git clone <this repo> && cd ai_infra_perf_prepare

# 0. one-time: passwordless ssh from this machine to every node
cp preflight/hosts.txt.example preflight/hosts.txt && vim preflight/hosts.txt
NODE_PASSWORD='...' bash tools/setup_ssh_keyless.sh preflight/hosts.txt --port 22

# 1. node consistency  (runs here, piped to the nodes over ssh)
mkdir -p results/preflight
bash preflight/cluster_runtime_check.sh -H preflight/hosts.txt -P 22 -o results/preflight/runtime.csv
bash preflight/cluster_spec_check.sh    -H preflight/hosts.txt -P 22 -o results/preflight/spec.csv

#    ... first copy this repo to <NODE_DIR> on every node — see VENDOR_RUNBOOK.md Step 3

# 2. single-GPU compute, every card, all nodes
bash cublas_bench/cluster_cublas_percard.sh \
     -H preflight/hosts.txt -P 22 -u root \
     -i <IMAGE> -g H800 -s <NODE_DIR>/cublas_bench

# 3. NCCL interconnect  (autoconf inspects a node, so run it ON one)
ssh <node> "cd <NODE_DIR>/nccl_bench && bash nccltest_autoconf.sh --apply"
#    ... then start the container on every node and run from one — see nccl_bench/readme.md

# 4. RDMA point-to-point
bash perftest/rail_isolation_check.sh node01 node02   # first: it constrains the rest
bash perftest/perftest_pair.sh node01 node02
bash perftest/perftest_agg.sh  node01 node02

# 5. package everything for submission
bash tools/collect_results.sh --vendor <name> --gpu H800 --dir results
python3 tools/summarize.py results --vendor <name> --gpu H800
```

This is the short version — **[VENDOR_RUNBOOK.md](VENDOR_RUNBOOK.md)** is the
step-by-step to actually follow, including what each gate should produce, how to
read it, and the traps. The acceptance tables to fill in are in
**[SUBMISSION.md](SUBMISSION.md)**.

## Prerequisites

### Cluster

| Item | Requirement |
| --- | --- |
| OS | Ubuntu 24.04 LTS (other modern distros work; the scripts are POSIX shell) |
| NVIDIA driver | ≥ 570 |
| CUDA | ≥ 12.9 (Blackwell/GB parts want the CUDA 13 era) |
| Docker | ≥ 26, with the NVIDIA container toolkit |
| Docker root dir | On a data disk with > 100 GB free |
| GPU access | `nvidia-smi` works inside a container |
| Root ssh between nodes | Passwordless, **required for the multi-node NCCL test** (`mpirun`) |
| Shared storage | Optional. Not required by these scripts — but useful for collecting logs |
| Local NVMe / data disk | Mounted at a consistent path, ≥ 3 TB recommended |
| Interface names | **Identical across nodes** (e.g. all `bond0`), and RDMA device names likewise |
| PCIe ACS | Disabled on every node (`nccl_bench/close_acs.sh`), otherwise intra-node P2P bandwidth suffers |
| Persistence mode | `nvidia-smi -pm 1` |

### Control machine

The machine you run the drivers from needs `ssh`, `sshpass`, `tar`, `python3` and
`git-lfs` (see the note below). The gate-2 and gate-3 scripts also need `perftest`
(`ib_write_bw` etc.) on the **nodes**, not on the control machine.

> **This repository uses git-lfs.** `cublas_bench/GB300/gemm-memread/` contains
> ~400 MB of `.so` libraries tracked by git-lfs. Without `git-lfs` installed you
> will get pointer files instead of libraries. Install git-lfs first, or clone
> with `GIT_LFS_SKIP_SMUDGE=1` if you do not need that package.

## Container images

The tests run inside a CUDA container. Use an image matching your GPU generation,
or build one from `nccl_bench/Dockerfile`.

| GPU generation | Image generation to use |
| --- | --- |
| B200 / B300 / GB200 / GB300 | NVIDIA PyTorch 26.05 era (CUDA 13) |
| H100 / H200 / H800 / H20 / A800 / L20 | NVIDIA PyTorch 25.05 era (CUDA 12.9) |

Any CUDA image that already contains `nccl-tests` built with the `_mpi` binaries
works for gate 2. The `cublasMatmulBench_<GPU>` binaries for gate 1 ship **in this
repository** — they are not part of any image.

## Layout

```
preflight/        gate 0 — node-consistency collectors and comparators
cublas_bench/     gate 1 — per-GPU GEMM benchmark, binaries, baselines
nccl_bench/       gate 2 — NCCL collective tests (container + MPI)
perftest/         gate 3 — RDMA point-to-point and rail diagnostics
tools/            ssh setup, result packaging, summary generation
results/          created by the gates; what you submit
VENDOR_RUNBOOK.md full step-by-step procedure
SUBMISSION.md     acceptance tables and submission format
```

## What gets submitted

`tools/collect_results.sh` packages the four gates' raw logs plus an environment
snapshot and a manifest into a single tarball, and `tools/summarize.py` turns it
into a written `SUMMARY.md` with the pass/fail verdicts. See
[SUBMISSION.md](SUBMISSION.md) for the checklist and the tables to complete.

**A gate with no output has not been accepted.** The packaging script names any
missing gate rather than silently omitting it.
