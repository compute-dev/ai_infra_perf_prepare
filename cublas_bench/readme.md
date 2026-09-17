# cuBLAS GEMM Benchmark

Single-GPU GEMM TFLOPS benchmark for NVIDIA GPUs (B200, B300, GB200, GB300,
H100, H200, H800, A800, H20, L20). Measures per-card compute throughput across
all supported precisions (FP4 / FP8 / INT8 / FP16 / BF16 / TF32 / FP32) and
compares against an official baseline.

## What this measures

- **Single-GPU GEMM TFLOPS only.** Each card is tested independently via
  `cublasMatmulBench`, a cuBLASLt matrix-multiply kernel at peak-saturating
  shapes. The result is a TFLOPS number per precision per card.

## What this does NOT measure

- **NVLink, RDMA, and topology.** This tool tests compute only. For
  interconnect bandwidth and topology validation, use `../nccl_bench/`.
- **Multi-GPU collective performance.** There is no NCCL or multi-GPU test here.

## Prerequisites

### Docker image

The `cublasMatmulBench_<GPU>` binaries ship **in this repo** and are **NOT** in
the image. The image only provides the CUDA runtime libraries.

| GPU family           | CUDA era  | Image example                         |
|----------------------|-----------|---------------------------------------|
| Blackwell (B200/B300)| CUDA 13+  | `nvcr.io/nvidia/pytorch:25.10-py3`    |
| Grace-Blackwell (GB200/GB300) | CUDA 13+ | `nvcr.io/nvidia/pytorch:25.10-py3` |
| Hopper (H100/H200/H800/A800) | CUDA 12 | `nvcr.io/nvidia/pytorch:25.05-py3` |
| Ampere (H20/L20)     | CUDA 12   | `nvcr.io/nvidia/pytorch:25.05-py3`    |

> The exact image does not matter as long as it has a matching CUDA runtime.
> Blackwell and Grace-Blackwell parts require CUDA 13+ era images; Hopper and
> Ampere work with CUDA 12 era images.

### Host requirements

- NVIDIA driver installed (`nvidia-smi` works)
- Docker with GPU support (`--gpus all`)
- The `cublas_bench/` directory accessible on each node (shared filesystem or
  copied locally)

## Quick start: single-node test

```bash
cd /path/to/cublas_bench

# Via the docker wrapper (recommended — handles CUDA libs)
GPU_NAME=H800 image=nvcr.io/nvidia/pytorch:25.05-py3 bash auto_cublas_bench_by_docker.sh

# Direct form (if CUDA runtime is already on the host)
GPU_NAME=H800 bash auto_cublas_bench.sh
```

Output is teed to `cublas_bench_<GPU>.log` and parsed live against
`cublasMatmulBenchBaseline.yaml`. Exit code 0 = all pass, 1 = some fail,
2 = usage error, 3 = the run produced no usable result (missing CUDA libs, or
nothing parseable in the log).

### The device-0 trap

The quick test **only exercises GPU 0**. `docker run --gpus all` merely exposes
all cards to the container; `cublasMatmulBench` has no `--device` argument, so
it always runs on device 0 (subject to `CUDA_VISIBLE_DEVICES`).

**"This node passed" does NOT mean all 8 cards passed.** Use the full-cluster
per-card sweep below for real acceptance.

## Full-cluster per-card sweep

### Overview

The cluster sweep runs `cublasMatmulBench` on **every GPU of every node**,
collects results, and performs fleet-wide outlier detection. This is the
correct way to accept a fleet: you are looking for **a card that reads
consistently low across several precisions** (cross-fleet outlier), not a
single precision that barely misses a baseline.

### Three scripts involved

| Script                   | Role                                                        |
|--------------------------|-------------------------------------------------------------|
| `cluster_cublas_percard.sh` | Orchestrator: SSH to each node, pipe the runner, collect  |
| `_percard_remote.sh`     | Remote runner: per-card loop on a single node (piped via stdin) |
| `parse_cluster_percard.py`| Aggregator: fleet median, outlier detection, CSV output   |

### Usage

```bash
# 1. Create a hosts file (one host per line, # comments allowed)
cat > hosts.txt <<'EOF'
node01
node02
node03
EOF

# 2. Preflight: check connectivity + image presence
bash cluster_cublas_percard.sh -H hosts.txt -i nvcr.io/nvidia/pytorch:25.05-py3 --preflight

# 3. Full sweep (interactive — will prompt for confirmation)
bash cluster_cublas_percard.sh -H hosts.txt -i nvcr.io/nvidia/pytorch:25.05-py3 -g H800

# 4. Non-interactive
bash cluster_cublas_percard.sh -H hosts.txt -i nvcr.io/nvidia/pytorch:25.05-py3 -g H800 -y
```

### Cluster script options

| Flag           | Default     | Description                                            |
|----------------|-------------|--------------------------------------------------------|
| `-H <file>`    | (required)  | Host list file                                         |
| `-i <image>`   | (required)  | Docker image                                           |
| `-P <port>`    | 22          | SSH port                                               |
| `-u <user>`    | root        | SSH user                                               |
| `-g <gpu>`     | H800        | GPU type                                               |
| `-s <dir>`     | $PWD        | Path to cublas_bench on remote nodes                   |
| `-n <n>`       | 8           | GPUs per node (auto-detect if _percard_remote runs without it) |
| `--preflight`  |             | Only do connectivity + image precheck                  |
| `-y`           |             | Non-interactive                                        |

### Transport

Plain SSH, parallelised via background jobs. No `pssh`, no `conda`, no jump
hosts, no shared NFS required for the runner script. The remote runner
`_percard_remote.sh` is **piped over stdin**:

```
ssh host 'SHARE_DIR=/path IMAGE=img GPU_TYPE=H800 bash -s' < _percard_remote.sh
```

The cublas binaries must be accessible at `SHARE_DIR` on each node (shared
filesystem or copied there beforehand).

### Outlier detection rationale

Judging a card is about **cross-fleet consistency**: a card that reads
consistently low across several precisions is a weak card. A single precision
barely missing the baseline is **not** a failure — it may be a shape or
algorithm artifact. The aggregator flags a card as an outlier only when one or
more of its precisions falls more than 3% below the **fleet median** for that
precision.

A cross-fleet spread of ~5–6% is normal. Within a single node, card-to-card
variation should be under ~3%.

## Second-opinion tool: torch_peak_probe.py

A companion tool that measures GEMM peak via PyTorch on the real framework path.
It **parses the sibling `cublas_bench_<GPU>.sh` at runtime** to get the exact
same m/n/k shapes — single source of truth, no hardcoded dimensions.

### Usage

```bash
# Dry run (no GPU/torch needed — just show the shapes it would test)
python3 torch_peak_probe.py --gpu H800 --dry-run

# Real run (inside a container with torch)
CUDA_VISIBLE_DEVICES=0 python3 torch_peak_probe.py --gpu H800

# Random input (reads ~8–12% lower than all-zeros)
CUDA_VISIBLE_DEVICES=0 python3 torch_peak_probe.py --gpu H800 --randn
```

### Two critical traps

1. **Random vs. zero input.** Random input reads ~8–12% lower than all-zeros.
   A comparison against cublas **MUST use `--zero`** (the default) to match the
   cublas `-B=0` runs. Never compare a `--randn` torch number against a
   `-B=0` cublas number.

2. **H800 TF32 via cublas `sss_fast_tf32` can read about half speed on some
   boards.** This is a **cublas tool artifact, NOT a weak card.** Real
   frameworks use `allow_tf32` (standard `COMPUTE_32F_TF32`), and the same
   board measures full speed there. **Never fail a card or RMA a board on the
   cublas `sss_fast_tf32` number alone** — judge TF32 on torch `allow_tf32`.
   Also note: **INT8 peak is authoritative from cublas (`bisb_imma`), not
   torch** — torch has no clean INT8 peak entry point, so `torch_peak_probe.py`
   skips INT8 entirely.

## The log-noise trap

When running inside Docker, you may see lines like:

```
libcompat_fix.so ... cannot be preloaded
```

These come from the host's `/etc/ld.so.preload` leaking into the container.
They are **unrelated to the benchmark result**. Filter them out:

```bash
grep -vE 'libcompat_fix|ld.so' cublas_bench_H800.log
```

## File reference

| File                          | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `auto_cublas_bench.sh`        | Single-node entry point: selects GPU script, runs, parses |
| `auto_cublas_bench_by_docker.sh` | Single-node entry point via Docker container           |
| `cublas_bench_<GPU>.sh`       | Per-GPU benchmark script (shapes and -P codes)           |
| `cublasMatmulBench_<GPU>`     | Pre-built binary for each GPU family                     |
| `cublasMatmulBenchBaseline.yaml` | Official TFLOPS baseline (GPU -> precision -> TFLOPS) |
| `parse_cublas_simple.py`      | Single-log parser: measures vs. baseline, exit code      |
| `cluster_cublas_percard.sh`   | Fleet orchestrator: SSH sweep, collect, aggregate        |
| `_percard_remote.sh`          | Remote runner: per-card loop on one node (stdin-piped)   |
| `parse_cluster_percard.py`    | Fleet aggregator: median, outliers, CSV                   |
| `torch_peak_probe.py`         | Second-opinion tool: torch GEMM peak on same shapes      |
| `diagnose_cuda.sh`            | CUDA runtime diagnostic (for missing-lib errors)         |
| `GB300/gemm-memread/`         | Separate CUTLASS-based GEMM+memread package (own docs)   |

## Troubleshooting

| Symptom                                              | Cause / Fix                                              |
|------------------------------------------------------|----------------------------------------------------------|
| `the input device is not a TTY`                      | Docker wrapper uses `-i` (not `-it`). If you need a TTY, set `DOCKER_TTY=1`. |
| `libcublasLt.so.12: cannot open shared object file` | CUDA runtime not in the container or host. Use a proper CUDA image, or run `diagnose_cuda.sh`. |
| `libcompat_fix.so ... cannot be preloaded`           | Host `/etc/ld.so.preload` leaking into container. Harmless; filter with `grep -vE 'libcompat_fix|ld.so'`. |
| Quick test passes but a card is actually slow         | Quick test only hits GPU 0. Run the full cluster per-card sweep. |
| H800 TF32 reads ~half speed                          | cublas `sss_fast_tf32` artifact. Judge TF32 on torch `allow_tf32`, not cublas. |
| `Error: image/IMAGE is required`                     | The Docker wrapper requires `-i <image>` or `image=<img>` env var. |
| `Error: -H <hosts_file> is required`                 | The cluster script requires `-H <file>` with a host list.  |
| Exit code 3                                          | The run produced no usable result — either CUDA shared libs are missing, or the log had nothing parseable. **This is not a pass.** |
| Exit code 1                                          | One or more precisions fell below baseline x 95%.        |
| `Result: NO RESULTS PARSED`                          | The benchmark never produced a timing — crashed, wrong container, or an error before the first result. Not a pass. |

## GB300/gemm-memread

`GB300/gemm-memread/` is a separate CUTLASS-based GEMM + memory-read benchmark
package with its own documentation in `GB300/gemm-memread/docs/README.md`. It
is **not part of this flow** — it is a standalone tool for GB300-specific
memory-bandwidth-aware GEMM measurements.
