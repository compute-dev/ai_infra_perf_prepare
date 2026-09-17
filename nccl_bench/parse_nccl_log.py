#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse NCCL Tests output and extract the peak out-of-place busbw (GB/s).

Works on the logs produced by run_nccltest.sh / 1node_nccltest.sh, e.g.
/root/nccltest_allreduce.log or /root/1node_nccltest_allgather.log.

NCCL Tests data lines look like:

       size       count    type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
        (B)  (elements)                              (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
   1073741824   268435456   float     sum      -1   8123.5   132.2   247.9      0   8120.1   132.2   247.9      0

Column layout for all_reduce (13 columns, 0-based):

    0: size(B)        5: time(us) OOP     8: #wrong OOP
    1: count          6: algbw OOP        9: time(us) IP
    2: type           7: busbw OOP  <--  10: algbw IP
    3: redop                             11: busbw IP  <--
    4: root                              12: #wrong IP

Different collectives have different column counts: all_gather and
reduce_scatter have no redop/root, alltoall has no redop. Rather than encode
each layout, this script reads from the RIGHT: the last 8 fields are always
out-of-place (time, algbw, busbw, #wrong) followed by in-place (time, algbw,
busbw, #wrong). So:

    OOP busbw = parts[-6]
    IP  busbw = parts[-2]

Usage:
    # one file
    python3 parse_nccl_log.py /root/nccltest_allreduce.log

    # several files
    python3 parse_nccl_log.py /root/nccltest_*.log

    # a directory (scans *nccltest_*.log)
    python3 parse_nccl_log.py /root/

    # only the peak numbers, one per line (for scripts)
    python3 parse_nccl_log.py /root/nccltest_allreduce.log --quiet

    # only the row at a given message size, which is what the baselines quote
    python3 parse_nccl_log.py /root/ --at-size 16G
"""

import argparse
import glob
import os
import re
import sys
from typing import List, Optional, Tuple

# A data row starts with a plain integer (byte size) and has at least 8 more
# whitespace-separated fields.
DATA_LINE_RE = re.compile(r"^\s*\d+\s+\d+\s+\S+.*$")

# run_nccltest.sh prints one of these before each collective. A combined log
# (the whole run tee'd to one file) therefore carries all four in sequence, and
# rows from different collectives must not be pooled together.
SECTION_RE = re.compile(r"^=+\s*(allreduce|all2all|allgather|reducescatter)\s*=+$", re.IGNORECASE)

# NCCL's own summary lines
AVG_BW_RE = re.compile(r"#\s*Avg\s+bus\s+bandwidth\s*:\s*([0-9.]+)", re.IGNORECASE)
OUT_OF_BOUNDS_RE = re.compile(r"#\s*Out\s+of\s+bounds\s+values\s*:\s*(\d+)", re.IGNORECASE)

# Anything that looks like a per-collective log file
LOG_GLOB = "*nccltest_*.log"

# Canonical report order for the four collectives
OP_ORDER = ["allreduce", "all2all", "allgather", "reducescatter"]



def human_size(num_bytes: int) -> str:
    """1073741824 -> '1.00 GB'"""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{num_bytes} B"


def parse_size(text: str) -> int:
    """Parse '16G' / '16384M' / '1073741824' into bytes."""
    m = re.fullmatch(r"\s*(\d+)\s*([kKmMgGtT]?)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid size: {text!r} (try 16G, 512M, 1073741824)")
    n = int(m.group(1))
    mult = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    return n * mult[m.group(2).lower()]


def parse_data_row(line: str) -> Optional[Tuple[int, float, float]]:
    """
    Parse one data row -> (size_bytes, oop_busbw, ip_busbw), or None.

    Reads the last 8 fields right-to-left: 4 out-of-place then 4 in-place
    (time, algbw, busbw, #wrong each).
    """
    parts = line.strip().split()
    # Need at least size count type + 8 timing fields
    if len(parts) < 11:
        return None

    try:
        size_bytes = int(parts[0])
        int(parts[1])
    except ValueError:
        return None

    try:
        oop_busbw = float(parts[-6])
        ip_busbw = float(parts[-2])
    except ValueError:
        return None

    return size_bytes, oop_busbw, ip_busbw


def parse_log(path: str) -> dict:
    """
    Parse one NCCL log file. Returns:
      {
        'file': path,
        'rows': [(size, oop_busbw, ip_busbw), ...],   # all rows, all sections
        'sections': {op: [(size, oop, ip), ...]},     # populated only for
                                                      # combined multi-test logs
        'peak_oop': (size, busbw) or None,
        'peak_ip':  (size, busbw) or None,
        'avg_busbw': float or None,     # NCCL's own summary
        'out_of_bounds': int or None,   # must be 0 in a clean run
      }
    """
    result = {
        "file": path,
        "rows": [],
        "sections": {},
        "peak_oop": None,
        "peak_ip": None,
        "avg_busbw": None,
        "out_of_bounds": None,
    }

    if not os.path.isfile(path):
        print(f"[WARN] no such file: {path}", file=sys.stderr)
        return result

    current_op = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue

            # Section marker — run_nccltest.sh prints these in combined logs
            m = SECTION_RE.match(stripped)
            if m:
                current_op = m.group(1).lower()
                result["sections"].setdefault(current_op, [])
                continue

            # Comment lines may carry the summary values
            if stripped.startswith("#"):
                m = AVG_BW_RE.search(stripped)
                if m:
                    try:
                        result["avg_busbw"] = float(m.group(1))
                    except ValueError:
                        pass
                m = OUT_OF_BOUNDS_RE.search(stripped)
                if m:
                    try:
                        # Keep the worst (largest) count seen across sections
                        val = int(m.group(1))
                        if result["out_of_bounds"] is None or val > result["out_of_bounds"]:
                            result["out_of_bounds"] = val
                    except ValueError:
                        pass
                continue

            row = parse_data_row(stripped)
            if row is None:
                continue
            result["rows"].append(row)
            if current_op is not None:
                result["sections"][current_op].append(row)

    if result["rows"]:
        peak_oop_row = max(result["rows"], key=lambda r: r[1])
        peak_ip_row = max(result["rows"], key=lambda r: r[2])
        result["peak_oop"] = (peak_oop_row[0], peak_oop_row[1])
        result["peak_ip"] = (peak_ip_row[0], peak_ip_row[2])

    return result


def expand_inputs(inputs: List[str]) -> List[str]:
    """Accept files, directories and globs."""
    files: List[str] = []
    for inp in inputs:
        if os.path.isdir(inp):
            files.extend(sorted(glob.glob(os.path.join(inp, LOG_GLOB))))
        elif any(c in inp for c in "*?[]"):
            files.extend(sorted(glob.glob(inp)))
        else:
            files.append(inp)
    # de-duplicate, preserving order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def row_at_size(rows, size_bytes: int) -> Optional[Tuple[int, float, float]]:
    """Return the row whose message size matches exactly, if present."""
    for size, oop, ip in rows:
        if size == size_bytes:
            return size, oop, ip
    return None


def extract(r: dict, at_size: Optional[int]) -> Optional[float]:
    """The number this report is about: at `at_size` if given, else the peak."""
    if at_size is not None:
        row = row_at_size(r["rows"], at_size)
        return row[1] if row else None
    return r["peak_oop"][1] if r["peak_oop"] else None


def print_report(results: List[dict], quiet: bool = False, at_size: Optional[int] = None) -> None:
    if quiet:
        for r in results:
            ops = [op for op in OP_ORDER if op in r["sections"]]
            if ops:
                # Combined log: one number per collective, in canonical order
                for op in ops:
                    row = row_at_size(r["sections"][op], at_size) if at_size is not None else None
                    if at_size is not None:
                        print(f"{row[1]:.2f}" if row else "N/A")
                    else:
                        sec_peak = max((x[1] for x in r["sections"][op]), default=None)
                        print(f"{sec_peak:.2f}" if sec_peak is not None else "N/A")
            else:
                val = extract(r, at_size)
                print(f"{val:.2f}" if val is not None else "N/A")
        return

    # When any input is a combined multi-test log, report one row per collective
    # per file — pooling rows across collectives would report a meaningless peak.
    if any(r["sections"] for r in results):
        label = "peak OOP busbw (GB/s)" if at_size is None else f"OOP busbw @{human_size(at_size)} (GB/s)"
        print(f"{label} — one row per collective")
        print()
        hdr = f"{'file':<40} {'collective':<14} {'busbw':>10} {'#oob':>6}"
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            fname = os.path.basename(r["file"])
            ops = [op for op in OP_ORDER if op in r["sections"]]
            if not ops:
                val = extract(r, at_size)
                oob = r["out_of_bounds"]
                print(f"{fname:<40} {'(whole file)':<14} "
                      f"{(f'{val:.2f}' if val is not None else 'N/A'):>10} "
                      f"{(str(oob) if oob is not None else '-'):>6}")
                continue
            for op in ops:
                row = row_at_size(r["sections"][op], at_size) if at_size is not None else None
                if at_size is not None:
                    val = row[1] if row else None
                else:
                    val = max((x[1] for x in r["sections"][op]), default=None)
                oob = r["out_of_bounds"]
                flag = "  <-- non-zero!" if oob else ""
                print(f"{fname:<40} {op:<14} "
                      f"{(f'{val:.2f}' if val is not None else 'N/A'):>10} "
                      f"{(str(oob) if oob is not None else '-'):>6}{flag}")
        print()
        print("Rows marked '#oob' report '# Out of bounds values'. It must be 0;")
        print("a non-zero value means the collective returned wrong results.")
        return

    # Per-collective log files
    label = "OOP busbw" if at_size is None else f"OOP busbw @{human_size(at_size)}"
    header = f"{'file':<46} {'rows':>6} {label:>20} {'@size':>10} {'peak IP':>10} {'avg':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        fname = os.path.basename(r["file"])
        nrows = len(r["rows"])

        if at_size is not None:
            row = row_at_size(r["rows"], at_size)
            val = f"{row[1]:.2f} GB/s" if row else "N/A"
            at = human_size(at_size) if row else "-"
        elif r["peak_oop"]:
            val = f"{r['peak_oop'][1]:.2f} GB/s"
            at = human_size(r["peak_oop"][0])
        else:
            val, at = "N/A", "-"

        peak_ip = f"{r['peak_ip'][1]:.2f}" if r["peak_ip"] else "N/A"
        avg = f"{r['avg_busbw']:.2f}" if r["avg_busbw"] is not None else "-"
        print(f"{fname:<46} {nrows:>6} {val:>20} {at:>10} {peak_ip:>10} {avg:>10}")



def main():
    ap = argparse.ArgumentParser(
        description="Parse NCCL Tests logs and extract the peak out-of-place busbw."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="log file(s) / directory / glob (e.g. /root/nccltest_*.log)",
    )
    ap.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="print only the extracted busbw numbers, one per line",
    )
    ap.add_argument(
        "--at-size",
        type=parse_size,
        metavar="SIZE",
        help="report the row at this message size instead of the peak "
             "(the baselines quote 16G). Accepts 16G / 512M / a byte count.",
    )
    ap.add_argument(
        "--csv",
        metavar="FILE",
        help="also dump every size->busbw row to a CSV file",
    )
    args = ap.parse_args()

    files = expand_inputs(args.inputs)
    if not files:
        print(f"[ERROR] no log files matched (looked for {LOG_GLOB})", file=sys.stderr)
        sys.exit(1)

    results = [parse_log(f) for f in files]
    print_report(results, quiet=args.quiet, at_size=args.at_size)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("file,size_bytes,oop_busbw_GBs,ip_busbw_GBs\n")
            for r in results:
                fname = os.path.basename(r["file"])
                for size, oop, ip in r["rows"]:
                    f.write(f"{fname},{size},{oop},{ip}\n")
        print(f"\n[INFO] details written to: {args.csv}")


if __name__ == "__main__":
    main()
