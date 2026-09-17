#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fleet-wide per-card cuBLAS aggregation and outlier detection.

Reads percard_cluster/<node>/dev{0..N}.log, builds a
(node, card) x precision matrix, and performs two checks:
  1) Baseline check: measured < official x (1 - TOL) is flagged as below baseline.
  2) Cross-fleet outlier: a card's reading for a precision that is below the
     fleet-wide median for that precision by more than OUTLIER_PCT.

Usage:
  python3 parse_cluster_percard.py [percard_cluster_dir] [GPU_TYPE]
Defaults: dir=./percard_cluster, GPU_TYPE=H800
"""
import os
import re
import sys
import glob
import statistics
import yaml

TOL = 0.05          # baseline tolerance (5%)
OUTLIER_PCT = 3.0   # more than this % below fleet median => outlier

PRECISIONS = ['FP8', 'INT8', 'FP16', 'TF32', 'FP32']
_ALL = 'FP4|FP8|INT8|FP16|BF16|TF32|FP32'
BASELINE_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'cublasMatmulBenchBaseline.yaml')


def extract(content):
    """Extract the first Gflops (-> TFLOPs) per precision from a single card's log."""
    row = {}
    for p in PRECISIONS:
        pat = (rf'\b({p})\b(?:(?!\b(?:{_ALL})\b).)*?Gflops\s*=\s*([0-9.]+)')
        m = re.search(pat, content, re.DOTALL)
        if m:
            row[p] = float(m.group(2)) / 1000.0
    return row


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'percard_cluster'
    gpu = (sys.argv[2] if len(sys.argv) > 2 else 'H800').upper()

    base = (yaml.safe_load(open(BASELINE_YAML)) or {}).get(gpu, {}) or {}

    # data[node][dev] = {prec: tflops}
    node_dirs = sorted(glob.glob(os.path.join(root, '*')),
                       key=lambda d: [int(x) for x in os.path.basename(d).split('.')
                                      if x.isdigit()] or [0])
    if not node_dirs:
        print(f"Error: no node directories under {root}", file=sys.stderr)
        sys.exit(2)

    data = {}
    for d in node_dirs:
        ip = os.path.basename(d)
        for logf in sorted(glob.glob(os.path.join(d, 'dev*.log'))):
            m = re.search(r'dev(\d+)\.log', logf)
            if not m:
                continue
            dev = int(m.group(1))
            row = extract(open(logf, encoding='utf-8', errors='ignore').read())
            if row:
                data.setdefault(ip, {})[dev] = row

    ncards = sum(len(v) for v in data.values())
    print(f"nodes={len(data)}  cards={ncards}  GPU_TYPE={gpu}  baseline={ {p: base.get(p) for p in PRECISIONS} }")

    # ---- fleet per-precision distribution ----
    fleet = {p: [] for p in PRECISIONS}
    for ip in data:
        for dev in data[ip]:
            for p, v in data[ip][dev].items():
                fleet[p].append(v)

    print("\n== fleet per-precision distribution ==")
    print(f"{'prec':<6}{'n':>4}{'min':>10}{'median':>10}{'max':>10}{'spread%':>9}{'base':>8}{'below95%':>14}")
    med = {}
    for p in PRECISIONS:
        vals = fleet[p]
        if not vals:
            continue
        mn, md, mx = min(vals), statistics.median(vals), max(vals)
        med[p] = md
        spread = (mx - mn) / md * 100 if md else 0
        b = base.get(p)
        nlow = sum(1 for v in vals if b and v < b * (1 - TOL))
        bstr = f"{b:.0f}" if b else "NA"
        print(f"{p:<6}{len(vals):>4}{mn:>10.1f}{md:>10.1f}{mx:>10.1f}{spread:>9.2f}{bstr:>8}{nlow:>14}")

    # ---- cross-fleet outlier cards: precision below fleet median > OUTLIER_PCT ----
    print(f"\n== cross-fleet outlier cards (precision < fleet median by {OUTLIER_PCT}%) ==")
    outliers = []
    for ip in data:
        for dev in sorted(data[ip]):
            for p in PRECISIONS:
                v = data[ip][dev].get(p)
                if v is None or p not in med or not med[p]:
                    continue
                dpct = (v - med[p]) / med[p] * 100
                if dpct < -OUTLIER_PCT:
                    outliers.append((ip, dev, p, v, med[p], dpct))
    if outliers:
        for ip, dev, p, v, m, dpct in outliers:
            print(f"  {ip} dev{dev} {p}: {v:.1f} vs median {m:.1f} ({dpct:+.2f}%)")
    else:
        print(f"  none (all cards within -{OUTLIER_PCT}% of fleet median for every precision)")

    # ---- per-node summary (median of that node's cards per precision) ----
    print("\n== per-node per-precision median ==")
    hdr = f"{'node':<16}" + "".join(f"{p:>9}" for p in PRECISIONS) + f"{'cards':>7}"
    print(hdr); print('-' * len(hdr))
    csv_lines = ["node,dev," + ",".join(PRECISIONS)]
    for ip in data:
        line = f"{ip:<16}"
        for p in PRECISIONS:
            vv = [data[ip][dev][p] for dev in data[ip] if p in data[ip][dev]]
            line += f"{statistics.median(vv):>9.1f}" if vv else f"{'--':>9}"
        line += f"{len(data[ip]):>7}"
        print(line)
        for dev in sorted(data[ip]):
            row = data[ip][dev]
            csv_lines.append(f"{ip},{dev}," + ",".join(
                f"{row[p]:.1f}" if p in row else "" for p in PRECISIONS))

    csv_path = os.path.join(root, 'fleet_percard.csv')
    open(csv_path, 'w').write("\n".join(csv_lines) + "\n")
    print(f"\nPer-card detail CSV: {csv_path}")

    # ---- conclusion ----
    print("\n== conclusion ==")
    if outliers:
        bad = sorted(set((ip, dev) for ip, dev, *_ in outliers))
        print(f"  {len(bad)} outlier card(s): {bad}")
        sys.exit(1)
    else:
        print("  No outlier cards: all nodes/cards are consistent across the fleet.")
        # Systematic baseline shortfall (all fleet low => not a bad card)
        for p in PRECISIONS:
            b = base.get(p)
            if b and med.get(p) and med[p] < b * (1 - TOL):
                print(f"  Note: {p} fleet median {med[p]:.1f} is systematically below "
                      f"baseline {b:.0f} ({(med[p]-b)/b*100:+.1f}%) — fleet-wide, not a bad card.")
        sys.exit(0)


if __name__ == '__main__':
    main()
