# Vendor runbook — step by step

Follow this top to bottom. Every gate writes into `results/<gate>/` and the
packaging script at the end collects all four. If a step produces no output,
that gate has **not** been accepted.

Replace these placeholders as you go:

| Placeholder | Meaning | Example |
| --- | --- | --- |
| `<IMAGE>` | CUDA container image (see [Choosing an image](#choosing-an-image)) | `nvcr.io/nvidia/pytorch:25.05-py3` |
| `<GPU>` | GPU model, matches the baseline tables | `H800`, `H20`, `B200` |
| `<NODE_DIR>` | Path where this repo lives on the **nodes** | `/data/ai_infra_perf_prepare` |
| `<PORT>` | SSH port of the nodes | `22` |
| `<USER>` | SSH user on the nodes | `root` |

Run everything from the **control machine** — the machine that can ssh to every
node — except where a step says otherwise.

---

## Step 0 — prepare the control machine

```bash
sudo bash install_env.sh
```

Installs `ssh`, `sshpass`, `rsync`, `python3`, `tar` and **`git-lfs`**. The
git-lfs part matters: `cublas_bench/GB300/gemm-memread/` is tracked with it, and
without it you get pointer files instead of libraries.

## Step 1 — write the node list

```bash
cp preflight/hosts.txt.example preflight/hosts.txt
vim preflight/hosts.txt
```

One entry per line, `#` comments allowed. Each entry is whatever **you** ssh to
the node with — a plain address (`10.0.0.11`) or an `~/.ssh/config` alias
(`node01`), in which case the port, user and any jump host come from that config
and you can drop the `-P` / `-u` arguments in the later steps. Aliases are the
easier option when the nodes are not directly reachable.

This list is used by gates 0, 1 and 3 and by the control loops in Step 6. Gate 2
needs a **second, different** list of addresses — see 6b.

## Step 2 — set up passwordless ssh

```bash
NODE_PASSWORD='<node password>' \
  bash tools/setup_ssh_keyless.sh preflight/hosts.txt --port <PORT> --user <USER>
```

Every line should print `[ OK ]`. Re-running is safe — the key is appended and
de-duplicated, never overwritten.

## Step 3 — distribute the repo to every node

Gates 1 and 2 need the repo on the nodes. Gate 0 and gate 3 do not (they pipe
scripts over ssh), but doing it once here keeps everything uniform.

```bash
NODES=$(grep -vE '^[[:space:]]*(#|$)' preflight/hosts.txt | awk '{print $1}')

for h in $NODES; do
  ssh -p <PORT> -o BatchMode=yes <USER>@$h "mkdir -p $(dirname <NODE_DIR>)"
  rsync -a --delete \
        --exclude '.git' --exclude 'cublas_bench/GB300' --exclude 'results' \
        --exclude '__pycache__' --exclude '*.log' --exclude '*.tar.gz' \
        ./ <USER>@$h:<NODE_DIR>/
  echo "$h done"
done
```

Check it:

```bash
ssh -p <PORT> <USER>@<first node> "ls <NODE_DIR>"
```

Reminder of the exact path you used — gates 1 and 2 take it as an argument.

---

## Step 4 — Gate 0: node consistency

**Why first:** `driver_version` matching does **not** mean the nodes match. Drift
here silently invalidates every later comparison.

```bash
mkdir -p results/preflight

bash preflight/cluster_runtime_check.sh \
     -H preflight/hosts.txt -P <PORT> -u <USER> \
     -o results/preflight/runtime.csv

bash preflight/cluster_spec_check.sh \
     -H preflight/hosts.txt -P <PORT> -u <USER> \
     -o results/preflight/spec.csv
```

**Expected:** both exit `0`, and `results/preflight/` contains two CSVs.

**If a critical (`★`) field differs → exit 2.** Exclude that node from
`preflight/hosts.txt`, re-run both, and record which nodes you excluded and why.
Do not drop a node silently.

> The compute-HCA set is auto-detected as "every HCA with an ACTIVE port", which
> can include a storage/management NIC. That is fine for a consistency check. If
> you want it scoped to the training fabric, set `COMPUTE_HCA_RE` and/or
> `EXCLUDE_HCA` — see `preflight/readme.md`.

---

## Step 5 — Gate 1: single-GPU compute (cuBLAS)

### 5a. Quick check first (one node, GPU 0 only)

Do this before committing to a fleet sweep — it validates the image and the
binary. Use the node the tests will run on, or any node:

```bash
cd cublas_bench
GPU_NAME=<GPU> image=<IMAGE> bash auto_cublas_bench_by_docker.sh
cd ..
```

**Expected:** a table of 6 precisions ending in `Result: ALL PASS`, exit `0`.

> This only exercises **GPU 0**. "This node passed" does not mean all 8 cards
> passed — that is what 5b is for.

### 5b. Full fleet, every card

```bash
bash cublas_bench/cluster_cublas_percard.sh \
     -H preflight/hosts.txt -P <PORT> -u <USER> \
     -i <IMAGE> -g <GPU> \
     -s <NODE_DIR>/cublas_bench
```

`-s` is the path to `cublas_bench` **on the nodes** — the path from Step 3. If
you get it wrong the run fails with `no_bench_script` immediately.

Runtime is roughly 5 minutes per card, so ~40 minutes for a node with 8 GPUs.
Use `--preflight` first if you want to check connectivity and the image without
running anything.

**Expected:** `results/cublas/percard_cluster/<node>/dev<N>.log` (one per card),
plus `fleet_percard.csv` and an outlier report.

**How to read it:** the headline is **cross-fleet consistency** — a card that
reads consistently low across several precisions is a weak card. A cross-fleet
spread of ~5–6 % is normal; within one node expect under ~3%. A single precision
slightly under baseline is usually a shape artifact, not a bad board.

> **Ignore `libcompat_fix.so ... cannot be preloaded`** in the logs — that is the
> host's `/etc/ld.so.preload` leaking into the container.
>
> **H800 TF32 via cuBLAS `sss_fast_tf32` can read about half speed on some
> boards. That is a cuBLAS artifact, not a weak card** — real frameworks use
> `allow_tf32`, where the same board is full speed. Judge H800 TF32 on torch
> `allow_tf32`, never on the `sss_fast_tf32` number.

---

## Step 6 — Gate 2: NCCL interconnect

### 6a. Derive the cluster-specific parameters

Run this **on one of the nodes**, not on the control machine — it inspects that
node's GPUs and RDMA devices.

```bash
FIRST=$(grep -vE '^[[:space:]]*(#|$)' preflight/hosts.txt | awk '{print $1}' | head -1)

ssh -p <PORT> <USER>@$FIRST \
  'cd <NODE_DIR>/nccl_bench && bash nccltest_autoconf.sh --apply'
```

Read the output and fix anything marked `[WARN]` before continuing. **Write down
`NETWORK_IF`, `NCCL_IB_HCA` and `NCCL_IB_GID_INDEX`** — you need them in 6c–6e.
`--apply` writes `set_env.sh` on the node.

### 6b. Write the **bootstrap** host list

There are **two different address spaces** here, and mixing them up is the
single most common way to get this gate wrong:

| List | Holds | Used by |
| --- | --- | --- |
| `preflight/hosts.txt` (Step 1) | addresses **you** ssh to | gates 0, 1, 3 and the control loops below |
| `nccl_bench/hosts.txt` (this step) | each node's address **on `NETWORK_IF`** | `mpirun` and NCCL, inside the container |

If your ssh path and your training network are the same, the two lists are
identical and you can just copy the file. If they differ — for example when you
reach the nodes through a bastion — they are **not**:

```bash
cp hosts.txt.example hosts.txt
vim hosts.txt          # one NETWORK_IF address per node, in the same order
```

> Being behind a bastion changes only the **ssh** list. Commands in 6c–6e
> therefore ssh to the nodes using `preflight/hosts.txt` (aliases work there,
> with the port and jump host coming from your ssh config), while the container
> is told to use `nccl_bench/hosts.txt`. Do not ssh to the bootstrap addresses.

### 6c. Start the container on every node

```bash
# ssh here uses the SSH list; the container's HOSTS_FILE uses the BOOTSTRAP list
SSH_NODES=$(grep -vE '^[[:space:]]*(#|$)' preflight/hosts.txt | awk '{print $1}')
IMAGE_NAME=<IMAGE>
NCCL_SOCKET_IFNAME=<NETWORK_IF from 6a>

for h in $SSH_NODES; do
  echo "=== $h ==="
  ssh <SSHOPTS> $h "mkdir -p <NODE_DIR>/nccl_bench"
  rsync -a -e '<SSHCMD>' nccl_bench/hosts.txt $h:<NODE_DIR>/nccl_bench/hosts.txt
  ssh <SSHOPTS> $h \
    "cd <NODE_DIR>/nccl_bench && \
     IMAGE_NAME=$IMAGE_NAME HOSTS_FILE=<NODE_DIR>/nccl_bench/hosts.txt \
     NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME bash launch_container.sh >/dev/null 2>&1 && \
     docker ps --format '{{.Names}}' | grep -q nccltest && echo 'container up'"
done
```

`<SSHOPTS>` is whatever your environment needs, e.g. `-p 22 -o BatchMode=yes`, or
nothing at all when your ssh config already carries the port and jump host.
`-e '<SSHCMD>'` is the matching `rsync` form, e.g. `-e "ssh -p 22"`.

**Expected:** `container up` for every node.

### 6d. Single-node NVLink test

On any one node (here, the first in the SSH list):

```bash
FIRST=$(grep -vE '^[[:space:]]*(#|$)' preflight/hosts.txt | awk '{print $1}' | head -1)

ssh <SSHOPTS> $FIRST \
  'docker exec -i nccltest bash -c "bash /root/hosts_dir/1node_nccltest.sh"'
```

Run this before the multi-node test: it isolates the NVLink domain from the
fabric, so a node-local problem cannot masquerade as a network problem.

**Expected:** each of the four collectives ends with `[<op>] exit=0` and
`# Out of bounds values : 0`.

### 6e. Multi-node test

```bash
ssh <SSHOPTS> $FIRST \
  'docker exec -i nccltest bash -c \
     "HOSTS_FILE=/root/hosts_dir/hosts.txt \
      NCCL_SOCKET_IFNAME=<NETWORK_IF> NCCL_IB_GID_INDEX=<GID index from 6a> \
      bash /root/run_nccltest.sh allreduce all2all allgather reducescatter"'
```

**Expected:** all four report success, each with `# Out of bounds values : 0`.

For other cluster sizes, edit `nccl_bench/hosts.txt` **on the node** (it is
bind-mounted, so the container picks the change up immediately) and re-run 6e —
no container restart needed:

```bash
ssh <SSHOPTS> $FIRST "vim <NODE_DIR>/nccl_bench/hosts.txt"
```

### 6f. Copy the logs out

Still on `$FIRST` — the logs are inside the container and are useless to the
packaging script until they are on the host:

```bash
ssh <SSHOPTS> $FIRST \
  'cd <NODE_DIR>/nccl_bench && bash collect_logs.sh'

# then bring them back to the control machine
rsync -a -e '<SSHCMD>' $FIRST:<NODE_DIR>/results/nccl/ results/nccl/
```

**Expected:** `results/nccl/` contains `nccltest_<op>.log` for each multi-node
collective and `1node_nccltest_<op>.log` for each single-node one.

### 6g. Stop the containers

```bash
for h in $SSH_NODES; do
  ssh <SSHOPTS> $h 'docker stop nccltest >/dev/null 2>&1; docker rm nccltest >/dev/null 2>&1'
done
```

**How to read the results:** compare the 16G peak against `nccl_bench/baselines.md`
at the same GPU / GPUs-per-node / node-count.

- **Cross-node All2All legitimately falls to ~10–20 % of the single-node
  figure** on every GPU generation. It is how the collective works — never judge
  multi-node All2All against a single-node number.
- **H800 AllReduce ≈215 GB/s is the reduced-NVLink hardware spec**, not a fault.
- On a rail-optimised fabric, only compare nodes on the same rail/leaf — Step 7
  tells you whether that constraint applies.

---

## Step 7 — Gate 3: RDMA point-to-point

Pick two nodes. `perftest` (`ib_write_bw` etc.) must be installed **on the
nodes**, and `ulimit -l` must be `unlimited` there.

### 7a. Is RDMA confined to one rail?

```bash
mkdir -p results/perftest
bash perftest/rail_isolation_check.sh <nodeA> <nodeB> --ssh-port <PORT> --ssh-user <USER> \
  | tee results/perftest/rail_isolation.txt
```

Pass the two nodes however you ssh to them (address or alias). Leave the device
arguments out — the script picks a GPU-affine compute rail automatically. If your
nodes name their rails differently, pass `devA devB` explicitly, otherwise you
will get a misleading verdict.

**Expected:** a verdict line. If it says **rail isolated**, then from this point
on only compare nodes on the same rail/leaf, and know that cross-node pipeline
parallelism is pairing-sensitive.

### 7b. Per-rail detail

```bash
bash perftest/perftest_pair.sh <nodeA> <nodeB> --ssh-port <PORT> --ssh-user <USER>
```

**Expected:** a per-rail table. A 200G port should read **~193 Gb/s** per single
QP, bidirectional roughly 2×, latency 2.5–2.9 µs. **All five error counters must
be zero** — grep the raw log for `hw_counters snapshot`.

> A single QP on a bonded rail is pinned to one member port by the LAG hash, so
> ~193 Gb/s on a 2×200G rail is correct, not half of a broken rail. 7c measures
> the real capacity.

### 7c. Per-rail aggregate

```bash
bash perftest/perftest_agg.sh <nodeA> <nodeB> --ssh-port <PORT> --ssh-user <USER>
```

**Expected:** on a 2×200G bonded rail, **~390 Gb/s peak**, not ~195. Judge on the
**peak** over ≥3 runs — a run that reads half is LAG hash luck, not a fault.

**Expected artefacts:** all three scripts write into `results/perftest/`
(`summary_<ts>.txt`, `summary_agg_<ts>.txt`, and the raw logs).

---

## Step 8 — Package and submit

```bash
bash tools/collect_results.sh --dir results --vendor <vendor name> --gpu <GPU>
python3 tools/summarize.py results --vendor <vendor name> --gpu <GPU>
```

`collect_results.sh` checks all four gate directories. **If any is empty it
refuses to package and names the gate** — that gate has not been accepted. It
then writes `results/environment.txt`, `results/MANIFEST.txt` and the tarball.

`summarize.py` writes `results/SUMMARY.md` and `results/results.json`, with a
verdict per gate and an overall verdict.

**Expected final state:**

```
results/
  SUMMARY.md          generated
  results.json        generated
  MANIFEST.txt        generated
  environment.txt     generated
  preflight/          runtime.csv, spec.csv
  cublas/             percard_cluster/<node>/dev<N>.log, fleet_percard.csv
  nccl/               nccltest_<op>.log, 1node_nccltest_<op>.log
  perftest/           summary_<ts>.txt, summary_agg_<ts>.txt, perftest_<ts>.log
```

Send the tarball plus the completed tables in **[SUBMISSION.md](SUBMISSION.md)**.

---

## Choosing an image

You need a CUDA container image. Either build one from `nccl_bench/Dockerfile`
(it adds nccl-tests and sshd), or use an NVIDIA PyTorch image of the right
generation:

| GPU generation | Image generation |
| --- | --- |
| B200 / B300 / GB200 / GB300 | CUDA 13 era (`nvcr.io/nvidia/pytorch:26.05-py3` or similar) |
| H100 / H200 / H800 / H20 / A800 / L20 | CUDA 12 era (`nvcr.io/nvidia/pytorch:25.05-py3` or similar) |

For gate 2 the image must already contain nccl-tests built with the `_mpi`
binaries (the `Dockerfile` here does that). The `cublasMatmulBench_<GPU>`
binaries for gate 1 ship **in this repo** and are never in the image.

---

## Things that will bite you

| Symptom | Cause |
| --- | --- |
| Gate 2: every rank prints `LOCALIP=NONE` | `NCCL_SOCKET_IFNAME` wrong, or not the same name on all nodes |
| Gate 2: `ibv_modify_qp INIT->RTR ETIMEDOUT` | Wrong `NCCL_IB_GID_INDEX`, or the RoCE QoS settings (MTU / PFC / trust) differ between nodes |
| Gate 2: AllGather/ReduceScatter crash with `all_gather.cu:50` on some pairs only | `NCCL_CUMEM_HOST_ENABLE=1` colliding with NVLS. The script defaults it to 0 — do not override |
| Gate 1: `no_bench_script` | `-s` points at the wrong path on the nodes — see Step 3 |
| Gate 3: every rail skipped as "no rail IP" | Should not happen; report it with `ibdev2netdev` output |
| Gate 3: `ibv_reg_mr` failed | `memlock` is not unlimited on that node |
| `collect_results.sh` says a gate is missing | That gate produced no files where expected — re-read its step above |

---

## What was verified where

Steps 0–8 were executed **as written in this document** against a two-node H20
cluster (8× H20 each, RoCE, GID index 3) on 2026-09-20. All four gates produced
their artefacts in `results/<gate>/` with no manual moving of files, and
`collect_results.sh` + `summarize.py` ran end to end.

Not exercised on that cluster, and worth a first-run check on yours:

- `tools/setup_ssh_keyless.sh` — passwordless ssh already existed there.
- `nccl_bench/close_acs.sh` — it changes PCIe configuration and needs a reboot,
  so it was deliberately not run.
- Gate 2 beyond two nodes, and gate 1 on a large fleet.
- Gate 1's fleet sweep was smoke-tested with one card per node rather than all
  eight; the mechanism is the same.
