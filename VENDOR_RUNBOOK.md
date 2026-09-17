# Vendor runbook

Step-by-step procedure for preparing a cluster and producing the acceptance
measurements. Each gate lists the exact commands, the artefact it should
produce, and the pass criterion.

If you only read one section, read [Before you start](#before-you-start) and
[How results are judged](#how-results-are-judged).

## Contents

1. [Before you start](#before-you-start)
2. [Gate 0 — node consistency](#gate-0--node-consistency)
3. [Gate 1 — single-GPU compute (cuBLAS)](#gate-1--single-gpu-compute-cublas)
4. [Gate 2 — NCCL interconnect](#gate-2--nccl-interconnect)
5. [Gate 3 — RDMA point-to-point](#gate-3--rdma-point-to-point)
6. [Packaging and submission](#packaging-and-submission)

---

## Before you start

### On the control machine

```bash
sudo bash install_env.sh          # ssh, sshpass, python3, git-lfs, tar
```

`git-lfs` matters: `cublas_bench/GB300/gemm-memread/` holds large `.so` files
under git-lfs. Without it you get pointer files, not libraries.

### On the cluster

| Requirement | Why | How to check |
| --- | --- | --- |
| `nvidia-smi` works on every node | everything depends on it | `nvidia-smi -L` |
| Docker + NVIDIA container toolkit | the tests run in a container | `docker run --rm --gpus all <cuda-image> nvidia-smi` |
| **PCIe ACS disabled** | ACS blocks peer-to-peer DMA, halving intra-node bandwidth | `bash nccl_bench/close_acs.sh` per node, then reboot |
| Persistence mode on | avoids clock ramp-up noise between runs | `nvidia-smi -pm 1` |
| 8 compute HCAs ACTIVE per node | the fabric must be fully cabled | `ls /sys/class/infiniband/`, check each `ports/1/state` is `4: ACTIVE` |
| Interface names identical fleet-wide | `mpirun` and NCCL bind by name | `ip -o -4 route show default` on several nodes |
| **memlock unlimited** (gate 3) | otherwise `ibv_reg_mr` fails and every RDMA test errors out | `ulimit -l` → should be `unlimited` |
| `perftest` installed (gate 3) | `ib_write_bw` etc. | `ib_write_bw --version` |
| Root ssh between nodes (gate 2) | `mpirun` spawns ranks over ssh | see below |

### Passwordless ssh

Gate 2 drives `mpirun`, which ssh's to each node **as root from inside the
container**. Gate 0 and gate 1 also assume you can reach the nodes.

```bash
cp preflight/hosts.txt.example preflight/hosts.txt
vim preflight/hosts.txt                       # one address per node

# This machine -> nodes (used by gates 0, 1 and 3)
NODE_PASSWORD='...' bash tools/setup_ssh_keyless.sh preflight/hosts.txt --port 22
```

Inside the container, gate 2 handles node-to-node ssh itself (the image ships a
keypair generated at build time), so you do not need to configure that.

Set `--port` / `--user` to match your environment throughout; every script takes
them. **Write them down** — a wrong port is the most common cause of "nothing
works".

---

## Gate 0 — node consistency

**Why first:** `driver_version` matching does **not** mean the nodes match. A
node with cuda-compat shadowing libcuda, a different ECC setting, a different
power limit, or one HCA down will still report the same driver version — and it
will quietly invalidate every cuBLAS and NCCL comparison that follows.

```bash
cd preflight
bash cluster_runtime_check.sh -H hosts.txt -P 22 -o ../results/preflight/runtime.csv
bash cluster_spec_check.sh    -H hosts.txt -P 22 -o ../results/preflight/spec.csv
```

**What it does:** collects ~35 runtime fields (B1–B8: driver stack, ECC, clocks
and power, the compute HCAs, the RDMA stack, kernel/OS, the bench image, and the
fabric — RoCE GID index, QoS, GPU-NIC rail topology, NVLink), takes the **most
common value** of each field across the fleet as the baseline, and flags nodes
that deviate.

**Pass criterion:** exit code 0. **Critical (`★`) fields differing → exit 2 →
FAIL.** Advisory fields only warn. An unreachable node also fails.

The two CSVs are the deliverable — they are your "the cluster is homogeneous"
evidence.

**If a node is unreachable or drifted:** exclude it explicitly, re-run, and note
the coverage (M/N nodes) in the report. Do not silently drop nodes — an
unstated exclusion is indistinguishable from a node that was never tested.

> Auto-detection nuance: by default the compute-HCA set is every HCA with an
> ACTIVE port. On a node with storage NICs that can sweep extra devices in.
> That is fine for a *consistency* check (the outlier still shows up), but if you
> want the check scoped to the training fabric, set `COMPUTE_HCA_RE` and/or
> `EXCLUDE_HCA`. See `preflight/readme.md`.

---

## Gate 1 — single-GPU compute (cuBLAS)

**What it proves:** every card reaches its expected GEMM throughput. This is how
you find an individual weak, throttled or degraded board — the kind of problem
that gets averaged away inside a full training run.

```bash
cd cublas_bench
```

### Quick check first (one node, one card)

```bash
GPU_NAME=H800 image=<cuda-image> bash auto_cublas_bench_by_docker.sh
```

This validates the image and the binary before you commit to a fleet sweep.

> **It only exercises GPU 0.** `--gpus all` exposes the cards; the binary has no
> device argument. "This node passed" does **not** mean all 8 cards passed.

### Full fleet, every card

```bash
bash cluster_cublas_percard.sh \
    -H ../preflight/hosts.txt -P 22 -u root \
    -i <cuda-image> -g H800
```

Runs every card of every node and aggregates with outlier detection.

**Artefacts:** `results/cublas/percard_cluster/<node>/dev<N>.log`, plus
`fleet_percard.csv` and an outlier report.

**Pass criterion — this is the part people get wrong:**

- The headline judgement is **cross-fleet consistency**, not "every precision
  beat the baseline". A card that reads **consistently low across several
  precisions** is a weak card.
- A cross-fleet spread of ~5–6 % is normal. Within one node it should be under
  ~3 %.
- A **single** precision slightly under baseline is often a shape or algorithm
  artifact, not a bad board. Investigate before escalating.

### Second opinion (optional but recommended)

`torch_peak_probe.py` measures the same shapes through PyTorch on the real
framework path, which distinguishes "this card is weak" from "this tool reads
this card oddly". Two traps, both of which have cost real time:

- **Use `--zero` (the default) to compare against cuBLAS.** Random input reads
  ~8–12 % lower, so a `--randn` run compared against a `-B=0` cuBLAS number
  looks like a failure when it is not.
- **H800 TF32 via cuBLAS `sss_fast_tf32` can read about half speed on some
  boards. This is a cuBLAS tool artifact, not a weak card** — real frameworks use
  `allow_tf32`, where the same board is full speed. Judge H800 TF32 on torch
  `allow_tf32`. Do not RMA a board on the `sss_fast_tf32` number.

Also ignore `libcompat_fix.so ... cannot be preloaded` lines in the log: that is
the host's `/etc/ld.so.preload` leaking into the container and is unrelated to
the result.

---

## Gate 2 — NCCL interconnect

**What it proves:** intra-node NVLink bandwidth and inter-node RDMA bandwidth
for the four collectives that matter, at several cluster sizes.

```bash
cd nccl_bench

# 1. derive every cluster-specific value (interface, HCA list, GID index, image)
bash nccltest_autoconf.sh -H hosts.txt
bash nccltest_autoconf.sh --apply          # writes set_env.sh

# 2. host lists
cp hosts.txt.example hosts.txt && vim hosts.txt
bash split_ip.sh                            # host8.txt, host16.txt, host32.txt, ...

# 3. start the container on EVERY node (see nccl_bench/readme.md for the loop)
# 4. from ONE node:
docker exec -i nccltest bash -c "bash /root/1node_nccltest.sh"
docker exec -i nccltest bash -c \
  "HOSTS_FILE=/root/hosts_dir/hosts.txt NCCL_SOCKET_IFNAME=<iface> \
   bash /root/run_nccltest.sh allreduce all2all allgather reducescatter \
   | tee /root/nccltest_all.log"

# 5. copy the logs out
mkdir -p ../results/nccl
docker cp nccltest:/root/. ../results/nccl/
```

Run the **single-node** test first: it isolates the NVLink domain from the
fabric, so a node-local problem does not masquerade as a network problem.

Run at **several scales** (1, 2, 4, 8, ... nodes). The bandwidth curve as a
function of scale is the interesting result — a flat curve that falls off at
some size points at spine oversubscription. Report the scale at which it bends.

**Pass criterion:** peak out-of-place busbw at the 16G message size, compared
against `nccl_bench/baselines.md` at the **same GPU / GPUs-per-node / node-count**
scale. ~90 % of the matching baseline is normal.

Two rules that prevent false alarms:

- **Cross-node All2All legitimately collapses to ~10–20 % of the single-node
  figure**, on every GPU generation. It is how the collective works. Never judge
  multi-node All2All against a single-node number.
- **H800 AllReduce ≈215 GB/s is the hardware spec** (reduced NVLink), not a fault.

Also confirm every log shows `# Out of bounds values : 0`. A non-zero count means
the collective returned **wrong results** — a correctness failure that matters
more than bandwidth.

---

## Gate 3 — RDMA point-to-point

**Why:** bandwith alone does not tell you whether the fabric is healthy. This
gate measures the raw RDMA path per rail and reads the hardware error counters,
which is how you separate a fabric fault from an NCCL/config fault.

```bash
cd perftest

# 1. Is RDMA confined to one rail? This constrains how gates 2 and beyond may
#    be grouped and compared — do it first.
bash rail_isolation_check.sh node01 node02

# 2. Per-rail detail, both directions, with error-counter snapshots
bash perftest_pair.sh node01 node02

# 3. Per-rail aggregate capacity (multi-QP, judged on peak over 3 runs)
bash perftest_agg.sh node01 node02
```

**Pass criterion:**

- A 200G port should read **~193 Gb/s** (~96.7 % of line rate) per single QP;
  400G ~389 Gb/s. Bidirectional write is roughly 2× unidirectional.
- Write and send latency **2.5–2.9 µs** on a healthy 200G rail.
- Aggregate on a 2×200G bonded rail: **~390 Gb/s peak**.
- **All five error counters must be zero** on every rail, before and after:
  `out_of_sequence`, `packet_seq_err`, `req_cqe_error`,
  `local_ack_timeout_err`, `rnr_nak_retry_err`.

Two things that look like faults but are not:

- **A single QP pinned to one member port of a bond.** On 802.3ad / LAG the hash
  fixes a given QP set to one physical port, so a single-QP reading of ~193 Gb/s
  on a 2×200G rail is correct. Use `perftest_agg.sh`, judge on the **peak**, and
  run it at least 3 times — a run that reads half is hash luck.
- **A trial that reports counters as `NA`.** That means the counter could not be
  read, so the rail is *unverified*, not verified-clean.

---

## Packaging and submission

```bash
bash tools/collect_results.sh --vendor <name> --gpu <model> --dir results
python3 tools/summarize.py results --vendor <name> --gpu <model>
```

`collect_results.sh` refuses to package if a gate produced nothing (override with
`--allow-partial` only if you intend to submit an incomplete package). It writes:

```
results/
  environment.txt    hardware/software snapshot of the collecting host
  MANIFEST.txt       what is present and what is missing
  SUMMARY.md         the written verdict (from summarize.py)
  results.json       the same, machine-readable
  <tarball>          everything above plus the raw logs
```

`summarize.py` re-runs the **same parsers** the gates use, so the summary can
never disagree with the raw data. Its overall verdict is `PASS`, `FAIL`,
`PASS (see review notes)`, or `INCOMPLETE` when any gate is missing.

Then complete the acceptance tables in **[SUBMISSION.md](SUBMISSION.md)** and send
the tarball.

---

## How results are judged

The single principle behind every gate: **compare like with like.**

| Do | Don't |
| --- | --- |
| Compare a fleet against its own median to find weak cards | Compare one card's single precision against a headline number |
| Compare multi-node All2All against a multi-node baseline | Compare it against the single-node number |
| Compare nodes on the same rail/leaf | Mix rails and assume the spread is noise |
| Report coverage (M of N nodes) and name excluded nodes | Silently drop an unreachable node |
| Treat "no output" as "not tested" | Treat it as a pass |

If a measurement is inconclusive, say so and say why. An honest gap is
recoverable; a wrong number that gets built on is not.
