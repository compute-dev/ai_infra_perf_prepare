# nccl_bench — NCCL collective / interconnect bandwidth test

**English** | [中文](readme.zh.md)

Measures intra-node NVLink and inter-node RDMA bandwidth with
[NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests), covering the four
collectives that matter for training: **AllReduce, All2All, AllGather,
ReduceScatter**.

This is gate 3 of the acceptance flow. Run it after [cuBLAS](../cublas_bench/readme.md)
(single-GPU compute) so that a compute problem is not mistaken for a fabric one.

## What it produces

`nccltest_<test>.log` per collective, plus a one-line summary per test of the
**peak out-of-place busbw at 16G**, which is the number to compare against
[baselines.md](baselines.md).

## Prerequisites

- Docker and the NVIDIA container runtime on every node.
- **Root ssh between nodes, no password** — background: `mpirun` spawns ranks
  over ssh. Set it up with [`../tools/setup_ssh_keyless.sh`](../tools/setup_ssh_keyless.sh).
- **PCIe ACS disabled** on every node (`close_acs.sh`), otherwise peer-to-peer
  DMA inside a node is blocked and intra-node bandwidth is low.
- A node list where every node can reach every other node, and the same
  interface name for the bootstrap network on all of them.
- Either an image with nccl-tests already built (`Dockerfile` here builds one),
  or a CUDA image that has it.

## Step 1 — detect the parameters

Almost every value these scripts need is cluster-specific. Do not guess them:

```bash
cd nccl_bench
bash nccltest_autoconf.sh -H hosts.txt      # -H also checks node reachability
bash nccltest_autoconf.sh --apply           # write the values into set_env.sh
```

It reports GPUs per node, the bootstrap interface, the RDMA HCA list (filtered to
the ones actually affine to a GPU), the fabric type, the RoCE v2 GID index, and
the image. Anything it could not determine is flagged `[WARN]` — fix those first.

> **The GID index is a real trap.** It is assigned by the driver and shifts when
> the IP scheme or RoCE version changes. A wrong value makes every QP fail with
> `ibv_modify_qp INIT->RTR ETIMEDOUT`, which reads like a fabric fault. Inspect
> it any time with `bash ../tools/detect_roce_gid.sh <dev> --table`.

## Step 2 — prepare the nodes

```bash
cp hosts.txt.example hosts.txt && vim hosts.txt   # one address per node
bash split_ip.sh                                  # host8.txt, host16.txt, host32.txt, ...
```

`split_ip.sh` writes one host list per power-of-two step so you can read the
bandwidth curve as a function of scale.

Disable ACS on every node (host change, reboot afterwards):

```bash
for h in $(cat hosts.txt); do ssh $h 'bash -s' < close_acs.sh; done
```

## Step 3 — start the container on every node

```bash
for h in $(cat hosts.txt); do
  ssh $h 'mkdir -p /root/nccl_bench'
  scp launch_container.sh run.sh run_nccltest.sh 1node_nccltest.sh \
      parse_nccl_log.py gpu_nic_topo.sh hosts.txt $h:/root/nccl_bench/
  ssh $h "cd /root/nccl_bench && IMAGE_NAME=<your image> HOSTS_FILE=/root/nccl_bench/hosts.txt \
          NCCL_SOCKET_IFNAME=<iface> bash launch_container.sh"
done
```

The container is privileged, sees all GPUs and `/dev/infiniband`, uses the host
network, and runs sshd on port 7900 so `mpirun` can reach peer nodes.

## Step 4 — run

On **one** node. Start with the single-node check — it isolates the NVLink domain
from the fabric, so a node-local problem does not masquerade as a network one.

```bash
# single node, all four collectives
docker exec -i nccltest bash -c "bash /root/1node_nccltest.sh"

# multi-node, all four collectives
docker exec -i nccltest bash -c \
  "HOSTS_FILE=/root/hosts_dir/hosts.txt \
   NCCL_SOCKET_IFNAME=<iface> NCCL_IB_GID_INDEX=<idx> \
   bash /root/run_nccltest.sh allreduce all2all allgather reducescatter \
   | tee /root/nccltest_all.log"
```

Other scales need no container restart — just point the host list the container
sees at a different size and re-run:

```bash
cp host16.txt hosts.txt && docker exec -i nccltest bash -c "bash /root/run_nccltest.sh all2all"
```

Individual collectives and sizes:

```bash
docker exec -i nccltest bash -c "bash /root/run_nccltest.sh allreduce"
docker exec -i nccltest bash -c "bash /root/run_nccltest.sh all2all b=8 e=4G n=20"
```

## Step 5 — parse

```bash
docker exec -i nccltest python3 /root/parse_nccl_log.py /root/                  # peak
docker exec -i nccltest python3 /root/parse_nccl_log.py /root/ --at-size 16G    # the baseline row
docker exec -i nccltest python3 /root/parse_nccl_log.py /root/ --csv /root/nccl.csv
```

Compare against [baselines.md](baselines.md), and also confirm every log shows
`# Out of bounds values : 0` — a non-zero count means the collective returned
wrong results, which matters more than bandwidth.

## Step 6 — collect the logs

The logs live **inside the container**. They are invisible to
`tools/collect_results.sh` until they are on the host, so copy them out on the
node where the tests ran:

```bash
bash collect_logs.sh                          # -> ../results/nccl/
bash collect_logs.sh --out /tmp/nccl_logs     # or wherever you like
```

If you ran the tests on a node and want the logs back on the control machine:

```bash
scp -P <PORT> -r <USER>@<node>:/data/ai_infra_perf_prepare/results/nccl/* results/nccl/
```

**Output:** `results/nccl/nccltest_<op>.log` for the multi-node run and
`1node_nccltest_<op>.log` for the single-node one. That is the directory the
packaging step expects.

## Step 7 — stop the containers

```bash
for h in $(cat hosts.txt); do
  ssh -p <PORT> <USER>@$h 'docker stop nccltest >/dev/null 2>&1; docker rm nccltest >/dev/null 2>&1'
done
```

## Multi-node judgement rules

- **All2All drops to ~10–20 % when it crosses nodes on every GPU generation.**
  This is how the collective works, not a defect. Compare multi-node All2All
  against a multi-node baseline, never against the single-node figure.
- **H800 AllReduce ≈215 GB/s is the hardware spec** (cut-down NVLink). Do not
  report it as a fault.
- On **rail-optimised fabrics**, only compare nodes **on the same rail/leaf**.
  Verify with [`../perftest/rail_isolation_check.sh`](../perftest/rail_isolation_check.sh)
  first; mixing rails makes the comparison meaningless.
- A **large-scale** drop (e.g. flat to 32 nodes, then falling) points at spine
  oversubscription. Report the scale at which it bends.

## Files

| File | Purpose |
| --- | --- |
| `nccltest_autoconf.sh` | Detect all cluster-specific parameters; `--apply` writes `set_env.sh` |
| `launch_container.sh` | Start the `nccltest` container on one node |
| `run.sh` | Container entrypoint: ssh config + sshd |
| `run_nccltest.sh` | Multi-node runner (mpirun, one process per GPU) |
| `1node_nccltest.sh` | Single-node runner (one process, all local GPUs) |
| `gpu_nic_topo.sh` | GPU ↔ HCA ↔ netdev affinity; produces `NCCL_IB_HCA` |
| `close_acs.sh` | Disable PCIe ACS (host-level, needs root) |
| `split_ip.sh` | Generate power-of-two host lists |
| `parse_nccl_log.py` | Extract peak / at-size busbw from logs |
| `collect_logs.sh` | Copy the logs out of the container into `results/nccl/` |
| `baselines.md` | Reference busbw values and judgement rules |
| `Dockerfile` | Builds an image with nccl-tests + sshd |
| `set_env.example.sh` | Config template |
| `hosts.txt.example` | Node list template |

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Every rank prints `LOCALIP=NONE` | `NCCL_SOCKET_IFNAME` wrong or not the same on all nodes |
| `ibv_modify_qp INIT->RTR ETIMEDOUT` | Wrong `NCCL_IB_GID_INDEX`, or RoCE QoS (MTU / PFC / trust) differs between nodes |
| `mpirun -np` spawns one rank too many | `nvidia-smi -L` emitting a spurious line. `run_nccltest.sh` counts `^GPU` lines to avoid this |
| AllGather / ReduceScatter crash with `unhandled system error (all_gather.cu:50)` on some pairs only | `NCCL_CUMEM_HOST_ENABLE=1` colliding with NVLS. The script defaults it to 0 — do not override |
| Single-node bandwidth well below baseline | PCIe ACS still enabled, or fewer GPUs visible than expected |
| Multi-node good on one pair, bad on another | Rail/leaf placement. Check with `rail_isolation_check.sh` |
| `mpirun` hangs or `ssh` inside the container fails | Container sshd not running, or port 7900 blocked between nodes |

## Note on ACS

`close_acs.sh` modifies PCIe configuration space and needs root and a reboot to
take full effect. It is a host-level change — make sure it is acceptable in your
environment before running it fleet-wide.
