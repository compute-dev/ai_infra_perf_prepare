# NCCL test baselines

Reference busbw values for the four collectives, measured the same way this
repo measures them, so you can tell "healthy" from "degraded".

## Measurement definition

- **Metric**: peak out-of-place busbw (GB/s) of the **16G** row.
- **Parameters**: `-b 8 -e 16G -f 2 -g 1 -n 50` — one process per GPU, 50
  iterations per size. This is exactly the default of `run_nccltest.sh`.
- **Tests**: AllReduce (`all_reduce_perf_mpi`), All2All (`alltoall_perf_mpi`),
  AllGather (`all_gather_perf_mpi`), ReduceScatter (`reduce_scatter_perf_mpi`).
- Extract the value with `parse_nccl_log.py --at-size 16G`.

**Pass criterion**: compare like with like — same GPU model × GPUs per node ×
node count. A result at roughly **≥90 % of the matching baseline** is normal.

## How to read these numbers

> **Single node (`*1`)** measures the intra-node domain (NVLink / NVSwitch). It
> is the machine's internal interconnect ceiling and the most useful first check.
>
> **Multi-node (`*2` and up)** measures the inter-node fabric (IB / RoCE).
> **All2All collapses across nodes** — that is expected, not a fault. See below.

## Single node — the first-line health check

| GPU | Scale | AllReduce | All2All | AllGather | ReduceScatter | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B300 | *8*1 | 845.6 | 643.4 | 671.5 | 691.9 | |
| B200 | *8*1 | 828 | 642 | 663 | 689 | CX7 |
| H100 | *8*1 | 479.1 | 348.1 | 361.2 | 365.5 | |
| H200 | *8*1 | 479.8 | 346.2 | 360.0 | 364.8 | |
| H20 | *8*1 | 483.3 | 293.7 | 359.2 | 364.9 | |
| H20 | *8*1 | 479.8 | 292.2 | 357.9 | 363.0 | second cluster, same tier |
| H800 | *8*1 | 215 | 143 | 164 | 165 | CX7 |
| A800 | *8*1 | 156.5 | 147.6 | 152.2 | 154.6 | 40GB |
| GB200 | *4*1 | 690.5 | 627.7 | 644.6 | 673.5 | |
| GB300 | *4*18 | 911 | 624 | 668 | 697 | smallest published scale |

**Reading notes:**

- Blackwell and GB parts sit at 690–910 GB/s AllReduce on a single node, roughly
  twice Hopper.
- H100 / H200 / H20 all land near 480 GB/s AllReduce — same NVLink4/NVSwitch
  generation, so intra-node bandwidth is close. They differ mainly in All2All
  (H100/H200 ≈347, H20 ≈293).
- **H800's AllReduce ≈215 is the hardware spec, not a fault.** Its NVLink is
  deliberately cut down. Do not raise a ticket against this number.

## Full table, including scale-out

Scale is written `GPU × GPUs_per_node × nodes`.

### Blackwell / GB

| Source | Scale | AllReduce | All2All | AllGather | ReduceScatter |
| --- | --- | ---: | ---: | ---: | ---: |
| B200 | B200*8*1 | 828 | 642 | 663 | 689 |
| B200 | B200*8*2 | 732 | 92 | 365 | 378 |
| B200 | B200*8*3 | 371 | 68 | 356 | 371 |
| B200 | B200*8*4 | 374 | 62 | 366 | 375 |
| B300 | B300*8*1 | 845.6 | 643.4 | 671.5 | 691.9 |
| B300 | B300*8*2 | 877.9 | 86.1 | 675.8 | 692.9 |
| B300 | B300*8*8 | 879.8 | 86.1 | 675.5 | 686.7 |
| B300 | B300*8*16 | 662.9 | 79.6 | 662.3 | 675.9 |
| B300 | B300*8*32 | 661.3 | 58.4 | 660.1 | 670.9 |
| B300 | B300*8*64 | 654.2 | 56.8 | 656.6 | 654.5 |
| B300 | B300*8*93 | 631.5 | 56.5 | 605.7 | 654.7 |
| B300 | B300*8*95 | 624.0 | 56.1 | 605.6 | 652.3 |
| GB200 | GB200*4*1 | 690.5 | 627.7 | 644.6 | 673.5 |
| GB200 | GB200*4*4 | 910.3 | 680.3 | 690.7 | 705.6 |
| GB200 | GB200*4*16 | 925.5 | 668.2 | 694.1 | 708.8 |
| GB200 | GB200*4*32 | 799.9 | 60.1 | 448.6 | 467.9 |
| GB300 | GB300*4*18 | 911 | 624 | 668 | 697 |
| GB300 | GB300*4*36 | 816 | 72 | 482 | 469 |
| GB300 | GB300*4*64 | 786 | 113 | 679 | 696 |
| GB300 | GB300*4*72 | 776 | 88 | 676 | 692 |
| GB300 | GB300*4*144 | 772 | 82 | 647 | 661 |

### Hopper (H100 / H200 / H20 / H800)

| Source | Scale | AllReduce | All2All | AllGather | ReduceScatter | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| H100 | H100*8*1 | 479.1 | 348.1 | 361.2 | 365.5 | |
| H100 | H100*8*2 | 472.9 | 51.0 | 313.9 | 355.6 | |
| H200 | H200*8*1 | 479.8 | 346.2 | 360.0 | 364.8 | |
| H200 | H200*8*2 | 484.9 | 72.9 | 325.7 | 329.8 | |
| H20 | H20*8*1 | 483.3 | 293.7 | 359.2 | 364.9 | |
| H20 | H20*8*2 | 488.2 | 55.5 | 223.7 | 223.7 | |
| H20 | H20*8*1 | 479.8 | 292.2 | 357.9 | 363.0 | second cluster |
| H20 | H20*8*2 | 476.8 | 91.5 | 323.7 | 372.7 | second cluster |
| H800 | H800*8*1 | 215 | 143 | 164 | 165 | CX7 |
| H800 | H800*8*2 | 222 | 91 | 174 | 170 | |
| H800 | H800*8*1 | 215 | 143 | 164 | 165 | second cluster |
| H800 | H800*8*2 | 223 | 56 | 164 | 164 | needs `NCCL_CUMEM_HOST_ENABLE=0` and `NCCL_IB_QPS_PER_CONNECTION=4` |

### Ampere (A800)

| Source | Scale | AllReduce | All2All | AllGather | ReduceScatter |
| --- | --- | ---: | ---: | ---: | ---: |
| A800-40GB | A800*8*1 | 156.5 | 147.6 | 152.2 | 154.6 |
| A800-40GB | A800*8*2 | 22.5 | 4.8 | 24.3 | 22.6 |

## Why cross-node All2All collapses

All2All drops from hundreds of GB/s to tens the moment traffic crosses nodes.
This happens on every GPU generation:

| GPU | Single node | Multi-node | Drop |
| --- | ---: | ---: | --- |
| B200 | 642 | 92 (*2) | to ~14 % |
| B300 | 643 | 86 (*2) | to ~13 % |
| GB200 | 628 (*1) | 60 (*32) | to ~10 % |
| H100 | 348 | 51 (*2) | to ~15 % |
| H200 | 346 | 73 (*2) | to ~21 % |
| H20 | 294 | 55 (*2) | to ~19 % |
| H800 | 143 | 56–91 (*2) | to ~40–64 % |
| A800 | 148 | 4.8 (*2) | to ~3 % (very weak fabric) |

**Cause**: All2All has every rank exchange with every other rank. Once it spans
nodes, nearly all traffic lands on the IB/RoCE fabric and is bounded by a single
port's bandwidth. AllReduce / AllGather / ReduceScatter use ring or tree
topologies and degrade much more gently.

**Consequence**: never judge multi-node All2All against the single-node number.
Compare it against a multi-node baseline at the same scale.

## Throughput is not the whole story

Also check `# Out of bounds values : 0` in every log. NCCL prints this line at
the end of each run; a non-zero count means the collective produced wrong
results, which is a correctness failure regardless of bandwidth. `run_nccltest.sh`
gears toward bandwidth, so a correctness bug would otherwise go unnoticed.
`parse_nccl_log.py` surfaces the value via `out_of_bounds`.
