# Node-consistency preflight

This directory holds a small set of Bash scripts that verify every node in a
GPU cluster is **configuration-consistent** *before* any performance test
(cuBLAS / NCCL / bitwise) runs. Run them first; they are cheap and they catch
the silent drift that otherwise invalidates every later number.

---

## Why a preflight, and why it runs first

`nvidia-smi` showing the same `driver_version` on two nodes does **not** mean
those nodes match. The driver version is one line; the things that actually
decide whether a cuBLAS or NCCL run is apples-to-apples are dozens of
runtime-state fields that the driver version never surfaces:

- which userspace `libcuda.so` is really resolved (cuda-compat can preempt it),
- ECC mode and accumulated uncorrected errors,
- GPU clock cap and power cap (a throttled/weak card skews every cuBLAS baseline),
- compute HCA presence / ACTIVE / rate / firmware,
- OFED, FabricManager, peermem,
- kernel / OS / persistence / memlock / NUMA,
- RoCE lossless config (trust + PFC), GPU-NIC rail topology, NVLink.

Any one of these drifting on a single node silently invalidates the
comparison. The preflight collects all of them on every node, collapses
per-host multi-card values into a single cross-host-comparable scalar, then
takes the **per-field mode** as the baseline and flags the outliers. Run it
before the first kernel launch; fix or exclude the outliers; only then run
the bench.

---

## Files

| file | role |
|------|------|
| `collect_runtime_state.sh`   | single-node software/runtime-state collector (B1-B8) |
| `collect_server_spec.sh`     | single-node hardware-spec collector (POC acceptance) |
| `cluster_runtime_check.sh`   | cluster consistency over the runtime-state fields |
| `cluster_spec_check.sh`      | cluster consistency over the hardware-spec fields |
| `compute_gpu_pciesw.sh`      | PCIe-switch topology per GPU (Gen / width / vendor) |
| `hosts.txt.example`          | commented host-list template |

All scripts are `chmod +x`'d.

---

## One-command entry points

```bash
# 1. Single node (run on any one node first, to sanity-check the collectors):
bash collect_runtime_state.sh          # human-readable table
bash collect_server_spec.sh            # human-readable spec table
bash collect_runtime_state.sh -j       # single-line JSON (what the cluster checker parses)
bash collect_server_spec.sh -j

# 2. Whole cluster (from a host that can SSH to every node on the list):
bash cluster_runtime_check.sh -H hosts.txt                      # default: root@...:22
bash cluster_runtime_check.sh -H hosts.txt -P 2222 -u admin      # custom port / user
bash cluster_runtime_check.sh -H hosts.txt -o rt.csv             # also write CSV

bash cluster_spec_check.sh -H hosts.txt                         # default: root@...:22, 64-way
bash cluster_spec_check.sh -H hosts.txt -p 128 -P 2222 -u admin # -p = parallelism, -P = ssh port
bash cluster_spec_check.sh -H hosts.txt -o spec.csv -k          # CSV + keep per-node raw JSON

# 3. PCIe-switch topology on one node:
bash compute_gpu_pciesw.sh            # summary table
bash compute_gpu_pciesw.sh -v         # verbose topology chain
```

Transport is **plain ssh only**. The collector is piped to each node's bash
over stdin (`ssh ... 'bash -s -- -j'`), so there is no shared filesystem, no
parallel-ssh fan-out tool, and no conda requirement. `hosts.txt` lists one
SSH-reachable address per line; `#` comments and blank lines are ignored.

### Output

The two CSVs **are** the gate-0 deliverable — they are the evidence that the
cluster is homogeneous. Write them into the results tree so the packaging step
finds them:

```bash
mkdir -p results/preflight
bash cluster_runtime_check.sh -H hosts.txt -P 22 -u root -o results/preflight/runtime.csv
bash cluster_spec_check.sh    -H hosts.txt -P 22 -u root -o results/preflight/spec.csv
```

`cluster_runtime_check.sh` exits `0` when all critical fields agree, `2` when a
critical field differs or a node is unreachable. Paths given to `-H` and `-o` are
relative to where you invoke the script.

---

## What the two collectors gather

### `collect_runtime_state.sh` -- the B1-B8 layers

| layer | what | why it matters |
|-------|------|-----------------|
| B1 | driver / CUDA-API / userspace libcuda / cuda-compat | bitwise + perf critical |
| B2 | ECC mode + cumulative uncorrected errors | marginal / degraded cards |
| B3 | GPU clock cap / power cap / throttling | cuBLAS weak-card baseline |
| B4 | each compute HCA present + ACTIVE + rate + firmware | multi-node link-up |
| B5 | OFED / fabricmanager / fabric state / peermem | multi-node & RDMA |
| B6 | kernel / OS / persistence / memlock / NUMA | debug + bitwise record |
| B7 | container bench image id | TFLOPs comparability |
| B8 | Fabric type / RoCE QoS / GPU-NIC topology / NVLink | pre-NCCL must-check |

Multi-card / multi-HCA values on one host are collapsed into one scalar:
all-identical -> that value; empty -> `N/A`; differing -> `mixed:v1|v2|..`.

### `collect_server_spec.sh` -- the spec layers

GPU (model x count), CPU (model + cores x sockets), Memory (type / per-DIMM /
per-socket / total), Storage-OS (root-disk capacity x count), Storage-Data
(data-disk capacity x count), NIC-compute (ConnectX model + measured link
rate x ports + chip-family spec cap), NIC-storage (BlueField/DPU x count).

> **JSON key asymmetry (intentional, do not "fix"):** `collect_server_spec.sh
> -j` emits `gpu_count`, `cpu_sockets` and `cores_per_socket` as **bare
> numbers**, while the string fields are quoted. The spec comparator only
> reads the quoted string fields. This is by design; leave it as is.

---

## How the comparators work

Both `cluster_*_check.sh` scripts share one algorithm:

1. Pipe the collector to every node over ssh, collect one single-line JSON
   per node.
2. For each field, take the **mode** (most frequent value across reachable
   nodes) as the baseline.
3. Flag every node whose value differs from the baseline.

### Runtime checker -- critical vs advisory (`cluster_runtime_check.sh`)

Fields marked **critical** (shown with `*` in the report) cause **exit 2** on
any diff:

`driver, cuda_api, libcuda, cuda_compat, gpu_count, ecc_mode, sm_clock_max,
power_limit, fabric_type, compute_hca, compute_active, compute_rate,
compute_mtu, compute_pcie, compute_fw, roce_gid_v2idx, roce_trust,
roce_pfc_prio, ofed, fabricmanager, fm_active, peermem, kernel,
gpu_nic_rail`

Advisory fields only warn (no FAIL):

`gpu_nic_topo, lldp_switch, nvlink_links, nvlink_speed, hca_devices, os,
persistence, memlock, numa_nodes, ecc_uncorr, bench_image_id`

(`ecc_uncorr` is advisory, but if every node agrees on a non-zero value it
still warns about possible ECC degradation.)

Exit codes: `0` = all critical fields agree; `2` = a critical field differs or
a node is unreachable; `1` = arg/dependency error.

### Spec checker (`cluster_spec_check.sh`)

Compares the seven quoted string fields (`gpu, cpu, memory, storage_os,
storage_data, nic_compute, nic_storage`). Any diff or any unreachable node ->
exit `2`; all identical -> exit `0`.

---

## Knobs: `COMPUTE_HCA_RE` / `EXCLUDE_HCA` / `BENCH_IMAGE_RE`

These are environment variables consumed by `collect_runtime_state.sh`. They
can be set on the command line or exported for the cluster checker.

### `COMPUTE_HCA_RE` -- scope the compute fabric

When **set**, it is used as an **anchored ERE** to select compute HCAs by
name, e.g.:

```bash
COMPUTE_HCA_RE='mlx5_10[1-8]' bash cluster_runtime_check.sh -H hosts.txt
```

When **unset** (the default), the script **auto-detects**: it takes every
device under `/sys/class/infiniband/` whose `ports/1/state` is `ACTIVE`, then
removes anything listed in `EXCLUDE_HCA`.

> **Storage-NIC trap.** An auto-detected set can include storage / management
> NICs alongside the training fabric. That is *acceptable for a consistency
> check* -- an outlier node is still caught -- but it compares the wrong
> population. When you want the check scoped to the training fabric only,
> **set `COMPUTE_HCA_RE`** explicitly.

### `EXCLUDE_HCA` -- drop names from the auto-detected set

Comma-separated **exact** HCA names, same convention as
`nccl_bench/gpu_nic_topo.sh`:

```bash
EXCLUDE_HCA=mlx5_1,mlx5_6 bash cluster_runtime_check.sh -H hosts.txt
```

Only consulted when `COMPUTE_HCA_RE` is unset.

### `BENCH_IMAGE_RE` -- scope the bench image

When **set**, `collect_runtime_state.sh` greps that pattern against
`docker images` to resolve `bench_image_id`. When **unset**, the field is
tagged `not-scoped` instead of matching any product or registry name -- no
image name is hardcoded:

```bash
BENCH_IMAGE_RE='mybench' bash cluster_runtime_check.sh -H hosts.txt
```

---

## How to read the two CSVs

`-o rt.csv` (runtime) and `-o spec.csv` (spec) each have one row per node:

```
host,reachable,<field1>,<field2>,...
```

- `host` -- the address from `hosts.txt`.
- `reachable` -- `1` if a JSON was collected, `0` if the node was unreachable
  or the collector failed.
- one column per field, holding the exact scalar the collector emitted
  (collapsed value, or `N/A`, or `mixed:...` for the runtime CSV).

Quick triage with shell tools:

```bash
# which nodes differ from the mode on a critical field?
awk -F, 'NR==1{for(i=1;i<=NF;i++)h[i]=$i} NR>1{print $1,$(diff_col)}' rt.csv

# reachable count
awk -F, 'NR>1{c[$2]++} END{for(k in c)print k,c[k]}' rt.csv
```

---

## Unreachable nodes are counted, never silently dropped

If a node produces no JSON (ssh failed, collector crashed), the checker marks
it `reachable=0`, lists it in the report under "unreachable nodes
(N/M)", and forces exit `2`. **Do not delete an unreachable node from
`hosts.txt` to make the check pass** -- leave it in so the gap is visible in
the report. Remove a node only when you genuinely intend to exclude it from
the audit.

For a non-root SSH user, `cluster_spec_check.sh` prefixes the remote command
with `sudo` so `dmidecode` / `lsblk` / `ethtool` fields stay complete
(passwordless sudo must be configured on the nodes).
