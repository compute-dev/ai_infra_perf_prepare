# perftest — RDMA point-to-point baseline

Measures the raw RDMA fabric between two nodes, independently of NCCL. This is
how you separate a **fabric** problem from an **NCCL / configuration** problem:
if point-to-point is at line rate, the fabric is fine and the fault is above it.

Complements [nccl_bench](../nccl_bench/readme.md), which measures what the
collectives actually achieve once NCCL sits on top of the same fabric.

## The four scripts

| Script | Question it answers |
| --- | --- |
| `perftest_pair.sh` | Per rail: write / read bandwidth, bidirectional bandwidth, write and send latency, and RDMA error counters. Finds a dead or degraded port. |
| `perftest_agg.sh` | Per rail: **aggregate** capacity using several QPs in one process. This is what NCCL uses. |
| `rail_isolation_check.sh` | Is RDMA confined to a single rail? Decides whether nodes may be compared and grouped freely. |
| `diag_gid.sh` | Is the RoCE GID index the reason connections fail? |

## Prerequisites

- **perftest** (`ib_write_bw`, `ib_read_bw`, `ib_write_lat`, `ib_send_bw`,
  `ib_send_lat`) on **both** nodes.
- **memlock unlimited** on both nodes (`ulimit -l`), otherwise `ibv_reg_mr`
  fails and every test errors out. Check with `ulimit -l`; set it in
  `/etc/security/limits.conf` (`* hard memlock unlimited`) and re-login.
- ssh from the machine you run these on to both nodes.

If perftest is not installed on the hosts, run these inside a container that has
it, started with `--network=host --privileged --device=/dev/infiniband
--ulimit memlock=-1`.

## Quick start

```bash
# 1. Is RDMA confined to one rail? Do this FIRST — it constrains everything below.
bash rail_isolation_check.sh node01 node02

# 2. Per-rail point-to-point detail, both directions, with counter snapshots
bash perftest_pair.sh node01 node02

# 3. Per-rail aggregate capacity (multi-QP, judged on peak over 3 runs)
bash perftest_agg.sh node01 node02
```

Devices are auto-discovered (every device with an ACTIVE port). Override with
`--devices "mlx5_0 mlx5_1"`; set the ssh port with `--ssh-port 2222`; set the
RoCE GID index with `--gid 3` (otherwise auto-detected).

## How to read the results

### Single QP (`perftest_pair.sh`)

Compare against the **link rate of one port**, not the rail:

- A 200G port should read **~193 Gb/s** (~24.2 GB/s, ≈96.7 % of line rate).
  Both directions should be the same.
- A 400G port should read ~389 Gb/s.
- Bidirectional write is roughly **2×** the unidirectional number.
- **Write and send latency: 2.5–2.9 µs** on a healthy 200G rail.

**On a bonded rail (802.3ad / mlx5 LAG), a single QP is pinned to one physical
port by the LAG hash.** So a single-QP reading of ~193 Gb/s on a 2×200G bond is
correct and expected — it is *not* half of a broken rail. Use
`perftest_agg.sh` for the rail's real capacity.

### Aggregate (`perftest_agg.sh`)

- A 2×200G bonded rail should reach **~390 Gb/s peak** (not ~195).
- **Judge on the peak, and run at least 3 times taking the maximum.** When all
  QPs hash onto one member port, a run legitimately reports about half. That is
  hash luck. A rail that stays at half across all runs is a genuine problem —
  check that member port's link state and cable.
- The `avg` will often be well below the `peak` on a bonded rail because the
  hash distributes QPs unevenly. Compare peaks.

> **Do not** derive "aggregate capacity" by summing the peaks of several
> independent short single-QP streams. That over-reports — it can exceed the
> physical ceiling of the rail, because the short runs never reach a coordinated
> steady state. `perftest_agg.sh` uses the correct method.

### Error counters — check these every time

`perftest_pair.sh` snapshots five counters per rail on both nodes, before the
run and after each direction:

```
out_of_sequence   packet_seq_err   req_cqe_error
local_ack_timeout_err   rnr_nak_retry_err
```

**All five must be 0 everywhere.** Any non-zero value identifies which node and
which rail is faulty. A rail can look fast on bandwidth and still be dropping or
reordering packets — that shows up as training instability later, not as low
bandwidth now. Grep the raw log for `hw_counters snapshot`.

### Rail isolation

| Same-rail | Cross-rail | Meaning | What to do |
| --- | --- | --- | --- |
| full rate | fails to connect | **rail isolated** | Compare bandwidth only within a rail/leaf; cross-node pipeline parallelism is pairing-sensitive; get an "HCA port → leaf" map from the network side |
| full rate | works | not rail isolated | Group freely; judge by leaf oversubscription |
| fails | — | the same-rail baseline did not run | Not a rail-isolation result — check device name, GID and cabling |

**Do not judge this by ping.** The IP layer is reachable across rails and will
mislead you; it is the RoCE path that is blocked.

> `rail_isolation_check.sh` deliberately swaps the **peer's `-d` device**, not
> just the IP address. Which rail RDMA takes is decided by each end's own `-d`;
> changing only the IP silently stays on the same rail and produces a false
> "cross-rail works". The script also verifies the peer server actually started,
> because a mistyped device name otherwise looks exactly like "rail isolated".

## Output

All three scripts write into `results/perftest/` by default, which is the
directory `tools/collect_results.sh` packages:

| File | From |
| --- | --- |
| `perftest_<ts>.log` | `perftest_pair.sh` — raw output plus the `hw_counters snapshot` blocks |
| `summary_<ts>.txt` | `perftest_pair.sh` — the per-rail table, one block per direction |
| `perftest_agg_<ts>.log` | `perftest_agg.sh` — raw output of every run |
| `summary_agg_<ts>.txt` | `perftest_agg.sh` — per-rail peak/avg over the runs |

Override with `--results DIR`. `rail_isolation_check.sh` prints to stdout only —
save it yourself if you want it in the bundle:

```bash
bash rail_isolation_check.sh node01 node02 --ssh-port 22 \
  | tee results/perftest/rail_isolation.txt
```

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `ibv_reg_mr` failed | memlock not unlimited on that node |
| `ibv_modify_qp INIT->RTR ETIMEDOUT` on both IB and RoCE | Wrong GID index — run `diag_gid.sh` |
| `Couldn't connect` / nothing listening | Server never started: wrong device name, or a leftover process on the port. The scripts use `pkill -x` (exact name); `pkill -f` would kill the ssh session itself |
| Single-QP ≈ half of the rail's nominal rate on a bond | Expected LAG behaviour — see above |
| Cross-rail works but is slow | Rail not isolated; likely leaf oversubscription |
| Bandwidth fine, but counters grow | Physical layer problem on that rail — cable, transceiver, or port |

## Files

| File | Purpose |
| --- | --- |
| `perftest_pair.sh` | Per-rail, both directions, with error-counter snapshots |
| `perftest_agg.sh` | Per-rail aggregate capacity, multi-QP, peak over N runs |
| `rail_isolation_check.sh` | Same-rail vs cross-rail verdict |
| `diag_gid.sh` | RoCE GID-index diagnostic (dumps the table on both nodes) |
