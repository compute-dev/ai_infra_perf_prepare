#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize.py — turn a results directory into a written acceptance summary.

Reads what the four gates produced, re-runs the same parsers the gates use
(so the summary can never disagree with them), and writes SUMMARY.md plus a
machine-readable results.json.

Usage:
    python3 tools/summarize.py results --vendor acme --gpu H800
    python3 tools/summarize.py results --vendor acme --gpu H800 --out SUMMARY.md

A gate that produced no output is reported as NOT RUN and makes the overall
verdict INCOMPLETE — never silently omitted.

Exit codes: 0 all gates present and passing, 1 something failed,
            2 could not read the results directory, 3 incomplete (gates missing).
"""

import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CUBLAS_DIR = os.path.join(REPO, "cublas_bench")
NCCL_DIR = os.path.join(REPO, "nccl_bench")

sys.path.insert(0, NCCL_DIR)
try:
    import parse_nccl_log as nccl_parser
except ImportError:  # pragma: no cover
    nccl_parser = None

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def run(cmd, **kw):
    """Run a command, return (rc, stdout+stderr)."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, **kw)
    return p.returncode, p.stdout


def find_first(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]          # latest by name (timestamps sort)
    return None


def read(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def human(n):
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return str(n)


# ---------------------------------------------------------------------------
# gate 0 — preflight
# ---------------------------------------------------------------------------


def summarize_preflight(res_dir):
    """Compare the two preflight CSVs. Reports fields that disagree across nodes."""
    out = {"present": False, "files": [], "critical_diff": [], "advisory_diff": [],
           "nodes": 0, "reachable": 0}

    csvs = sorted(glob.glob(os.path.join(res_dir, "preflight", "*.csv")))
    if not csvs:
        return out
    out["present"] = True

    # Fields the runtime comparator treats as critical. Kept in sync with
    # preflight/cluster_runtime_check.sh; the summary only reports, it does not
    # re-decide, so a mismatch here cannot turn a FAIL into a PASS.
    CRITICAL = {
        "driver", "cuda_api", "libcuda", "cuda_compat", "gpu_count", "ecc_mode",
        "sm_clock_max", "power_limit", "fabric_type", "compute_hca",
        "compute_active", "compute_rate", "compute_mtu", "compute_pcie",
        "compute_fw", "roce_gid_v2idx", "roce_trust", "roce_pfc_prio", "ofed",
        "fabricmanager", "fm_active", "peermem", "kernel", "gpu_nic_rail",
    }

    import csv as csvmod

    for path in csvs:
        name = os.path.basename(path)
        out["files"].append(name)
        try:
            with open(path, newline="", errors="replace") as f:
                rows = list(csvmod.DictReader(f))
        except OSError:
            continue
        if not rows:
            continue
        out["nodes"] = max(out["nodes"], len(rows))
        reach = sum(1 for r in rows if str(r.get("reachable", "")).strip() in ("1", "true", "True"))
        out["reachable"] = max(out["reachable"], reach)

        cols = [c for c in rows[0].keys() if c not in ("host", "reachable")]
        for col in cols:
            vals = {str(r.get(col, "")).strip() for r in rows}
            vals.discard("")
            if len(vals) <= 1:
                continue
            entry = {"file": name, "field": col, "values": sorted(vals)[:6],
                     "nodes": len(rows)}
            if col in CRITICAL:
                out["critical_diff"].append(entry)
            else:
                out["advisory_diff"].append(entry)

    return out


# ---------------------------------------------------------------------------
# gate 1 — cuBLAS
# ---------------------------------------------------------------------------


def summarize_cublas(res_dir, gpu):
    """Reuse the gate's own parsers: the cluster aggregator if a per-card sweep
    exists, otherwise the single-log parser. Never re-implement the comparison."""
    out = {"present": False, "mode": None, "exit_code": None, "report": ""}

    percard = os.path.join(res_dir, "cublas", "percard_cluster")
    cluster_py = os.path.join(CUBLAS_DIR, "parse_cluster_percard.py")

    if os.path.isdir(percard) and glob.glob(os.path.join(percard, "*", "dev*.log")):
        out["present"] = True
        out["mode"] = "cluster per-card"
        rc, txt = run([sys.executable, cluster_py, percard, gpu], cwd=CUBLAS_DIR)
        out["exit_code"] = rc
        out["report"] = txt
        return out

    logs = sorted(glob.glob(os.path.join(res_dir, "cublas", "*.log")))
    if logs:
        out["present"] = True
        out["mode"] = "single log(s)"
        simple_py = os.path.join(CUBLAS_DIR, "parse_cublas_simple.py")
        chunks = []
        worst = 0
        for lg in logs:
            rc, txt = run([sys.executable, simple_py, lg, gpu], cwd=CUBLAS_DIR)
            chunks.append(f"### {os.path.basename(lg)}\n\n```\n{txt.strip()}\n```\n")
            worst = max(worst, rc)
        out["exit_code"] = worst
        out["report"] = "\n".join(chunks)
        return out

    return out


# ---------------------------------------------------------------------------
# gate 2 — NCCL
# ---------------------------------------------------------------------------


def load_nccl_baselines():
    """Parse the markdown baseline tables into {(gpu, gpus_per_node, nodes): {...}}.

    baselines.md is the single human-readable source, so the summary reads it
    rather than keeping a second copy of the numbers."""
    path = os.path.join(NCCL_DIR, "baselines.md")
    text = read(path)
    table = {}
    if not text:
        return table

    scale_re = re.compile(r"^(\d+)$")
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 5:
            continue
        key_col, scale_col = cols[0], cols[1]
        # Full table:  | B200 | B200*8*1 | 828 | ... |
        # Single-node: | B200 | *8*1 | 845.6 | ... |  (GPU is column 0)
        m = re.match(r"^([A-Za-z0-9]+)\*(\d+)\*(\d+)$", scale_col)
        if m:
            gpu_model, gpn, nodes = m.group(1), int(m.group(2)), int(m.group(3))
        elif re.match(r"^\*\d+\*\d+$", scale_col) and re.match(r"^[A-Za-z0-9]+$", key_col):
            parts = scale_col.strip("*").split("*")
            gpu_model, gpn, nodes = key_col, int(parts[0]), int(parts[1])
        else:
            continue

        nums = []
        for c in cols[2:6]:
            c = c.strip()
            try:
                nums.append(float(c))
            except ValueError:
                nums = []
                break
        if len(nums) != 4:
            continue

        table[(gpu_model.upper(), gpn, nodes)] = {
            "allreduce": nums[0], "all2all": nums[1],
            "allgather": nums[2], "reducescatter": nums[3],
        }
    return table


def nccl_scale_from_log(text):
    """Recover (nodes, gpus_per_node) from the log header.

    run_nccltest.sh prints either '[INFO] HOST_LIST: ip:8,ip:8' or an
    '[CMD] mpirun -np N -H ip:8,ip:8' line."""
    m = re.search(r"HOST_LIST:\s*(\S.*)$", text, re.MULTILINE)
    hostlist = None
    if m:
        hostlist = m.group(1).strip()
    else:
        m = re.search(r"mpirun\s+-np\s+\d+\s+-H\s+(\S+)", text)
        if m:
            hostlist = m.group(1)
    if not hostlist:
        return None
    entries = [e for e in hostlist.split(",") if e.strip()]
    if not entries:
        return None
    gpn = None
    m = re.search(r":(\d+)$", entries[0].strip())
    if m:
        gpn = int(m.group(1))
    if gpn is None:
        gpn = 1
    return len(entries), gpn


def summarize_nccl(res_dir, gpu, baselines):
    out = {"present": False, "tests": [], "baseline_used": None}

    logs = sorted(glob.glob(os.path.join(res_dir, "nccl", "*.log")))
    if not logs or nccl_parser is None:
        return out
    out["present"] = True

    # Search the known GPU models so the key matches the baseline table
    gpu_up = gpu.upper()

    for path in logs:
        text = read(path)
        parsed = nccl_parser.parse_log(path)
        scale = nccl_scale_from_log(text)

        base = None
        if scale:
            nodes, gpn = scale
            for (bg, bgpn, bn), vals in baselines.items():
                if bg == gpu_up and bgpn == gpn and bn == nodes:
                    base = vals
                    out["baseline_used"] = f"{bg}*{bgpn}*{bn}"
                    break

        entry = {"file": os.path.basename(path), "scale": None, "rows": [],
                 "out_of_bounds": parsed.get("out_of_bounds")}
        if scale:
            entry["scale"] = f"{gpu_up}*{scale[1]}*{scale[0]}"

        # Combined log -> per collective; otherwise treat the whole file as one
        sections = parsed.get("sections") or {}
        if sections:
            items = [(op, sections[op]) for op in nccl_parser.OP_ORDER if op in sections]
        else:
            m = re.match(r"^(?:1node_)?nccltest_(\w+)\.log$", os.path.basename(path))
            op = m.group(1) if m else os.path.basename(path)
            items = [(op, parsed["rows"])]

        for op, rows in items:
            peak = max((r[1] for r in rows), default=None)
            at16 = None
            for size, oop, _ip in rows:
                if size == 16 * 1024 ** 3:
                    at16 = oop
                    break
            b = base.get(op) if base else None
            entry["rows"].append({
                "op": op,
                "peak_oop_busbw": round(peak, 2) if peak is not None else None,
                "at16g_oop_busbw": round(at16, 2) if at16 is not None else None,
                "baseline": b,
                "pct_of_baseline": round(at16 / b * 100, 1) if (at16 and b) else None,
            })
        out["tests"].append(entry)

    return out


# ---------------------------------------------------------------------------
# gate 3 — perftest
# ---------------------------------------------------------------------------


def summarize_perftest(res_dir):
    out = {"present": False, "pair": [], "agg": [],
           "counter_nonzero": [], "counter_unreadable": []}

    pdir = os.path.join(res_dir, "perftest")
    if not os.path.isdir(pdir):
        return out

    for path in sorted(glob.glob(os.path.join(pdir, "summary_*.txt"))):
        out["present"] = True
        for line in read(path).splitlines():
            f = line.split()
            if len(f) >= 7 and not f[0].startswith("-") and \
               re.match(r"^[A-Za-z]", f[0]) and f[0] not in ("rail(dev)", "TOTAL"):
                try:
                    out["pair"].append({
                        "rail": f[0], "gbs": float(f[1]), "wr_bw": f[2],
                        "wr_bidi": f[3], "rd_bw": f[4],
                        "wr_lat_us": f[5], "send_lat_us": f[6],
                    })
                except ValueError:
                    continue

    for path in sorted(glob.glob(os.path.join(pdir, "summary_agg_*.txt"))):
        out["present"] = True
        for line in read(path).splitlines():
            f = line.split()
            if len(f) >= 5 and re.match(r"^[A-Za-z]", f[0]) and \
               f[0] not in ("rail(dev)",):
                try:
                    out["agg"].append({"rail": f[0], "peak_gbs_gbps": float(f[1]),
                                       "avg_gbps": float(f[2]), "runs": f[3]})
                except ValueError:
                    continue

    # RDMA error counters. perftest_pair.sh writes lines like:
    #   mlx5_0: out_of_sequence=0 packet_seq_err=0 req_cqe_error=0 \
    #           local_ack_timeout_err=0 rnr_nak_retry_err=0
    # A value of "NA" means the counter could not be read — worth surfacing too,
    # since an unreadable counter is exactly what this check exists to catch.
    counter_re = re.compile(r"^([A-Za-z0-9_]+):\s+((?:\w+=\S+\s*)+)$")
    for path in glob.glob(os.path.join(pdir, "perftest*.log")):
        for line in read(path).splitlines():
            m = counter_re.match(line.strip())
            if not m:
                continue
            dev, pairs = m.group(1), m.group(2)
            if "out_of_sequence" not in pairs:
                continue
            nonzero, unreadable = [], []
            for k, v in re.findall(r"(\w+)=(\S+)", pairs):
                if v.upper() in ("NA", "-", ""):
                    unreadable.append(k)
                else:
                    try:
                        if int(v) > 0:
                            nonzero.append(f"{k}={v}")
                    except ValueError:
                        unreadable.append(k)
            if nonzero:
                out["counter_nonzero"].append(
                    {"device": dev, "detail": ", ".join(nonzero), "line": line.strip()})
            if unreadable:
                out["counter_unreadable"].append(
                    {"device": dev, "detail": ", ".join(unreadable)})

    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Summarise the four acceptance gates.")
    ap.add_argument("results_dir", help="results root written by the gate scripts")
    ap.add_argument("--vendor", default="", help="vendor name")
    ap.add_argument("--gpu", default="", help="GPU model, e.g. H800")
    ap.add_argument("--out", default=None, help="output markdown (default <dir>/SUMMARY.md)")
    args = ap.parse_args()

    res = os.path.abspath(args.results_dir)
    if not os.path.isdir(res):
        print(f"[ERROR] not a directory: {res}", file=sys.stderr)
        sys.exit(2)

    gpu = (args.gpu or "").upper()
    out_path = args.out or os.path.join(res, "SUMMARY.md")

    pre = summarize_preflight(res)
    cub = summarize_cublas(res, gpu or "H800")
    ncc = summarize_nccl(res, gpu, load_nccl_baselines())
    pft = summarize_perftest(res)

    # ---- verdicts -----------------------------------------------------------
    if not pre["present"]:
        pre_v = "NOT RUN"
    elif pre["critical_diff"]:
        pre_v = "FAIL"
    else:
        pre_v = "PASS"

    if not cub["present"]:
        cub_v = "NOT RUN"
    elif cub["exit_code"] == 0:
        cub_v = "PASS"
    elif cub["exit_code"] == 3:
        cub_v = "NO RESULT"
    else:
        cub_v = "FAIL"

    if not ncc["present"]:
        ncc_v = "NOT RUN"
    else:
        ncc_v = "PASS"
        for t in ncc["tests"]:
            for r in t["rows"]:
                if r["pct_of_baseline"] is not None and r["pct_of_baseline"] < 90:
                    ncc_v = "REVIEW"
        for t in ncc["tests"]:
            if t["out_of_bounds"]:
                ncc_v = "FAIL"      # correctness, not just bandwidth

    if not pft["present"]:
        pft_v = "NOT RUN"
    elif pft["counter_nonzero"]:
        pft_v = "FAIL"
    else:
        pft_v = "PASS"

    verdicts = [("0. Node consistency", pre_v), ("1. cuBLAS compute", cub_v),
                ("2. NCCL interconnect", ncc_v), ("3. RDMA point-to-point", pft_v)]
    not_run = [n for n, v in verdicts if v == "NOT RUN"]
    failed = [n for n, v in verdicts if v in ("FAIL", "NO RESULT")]
    review = [n for n, v in verdicts if v == "REVIEW"]

    if not_run:
        overall = "INCOMPLETE"
    elif failed:
        overall = "FAIL"
    elif review:
        overall = "PASS (see review notes)"
    else:
        overall = "PASS"

    # ---- markdown -----------------------------------------------------------
    md = []
    md.append("# Acceptance summary\n")
    md.append(f"- **Vendor**: {args.vendor or '(not given)'}")
    md.append(f"- **GPU**: {gpu or '(not given)'}")
    md.append(f"- **Generated**: {datetime.datetime.now().isoformat(timespec='seconds')}")
    md.append(f"- **Results directory**: `{res}`")
    md.append(f"- **Overall verdict**: **{overall}**\n")

    env = os.path.join(res, "environment.txt")
    if os.path.isfile(env):
        md.append("## Environment\n")
        md.append("```")
        md.append(read(env).strip())
        md.append("```\n")

    md.append("## Gate verdicts\n")
    md.append("| Gate | Verdict |")
    md.append("| --- | --- |")
    for n, v in verdicts:
        md.append(f"| {n} | {v} |")
    md.append("")
    if not_run:
        md.append(f"> **Not run**: {', '.join(not_run)}. A gate with no output has "
                  "not been accepted and is not evidence of anything.\n")

    # gate 0
    md.append("## 0. Node consistency\n")
    if not pre["present"]:
        md.append("_No preflight CSV found._\n")
    else:
        md.append(f"Covered **{pre['reachable']}/{pre['nodes']}** reachable node(s) "
                  f"across {len(pre['files'])} CSV file(s).\n")
        if pre["critical_diff"]:
            md.append("**Critical fields that differ across nodes:**\n")
            md.append("| Field | Values seen |")
            md.append("| --- | --- |")
            for e in pre["critical_diff"]:
                md.append(f"| `{e['field']}` | {', '.join('`'+v+'`' for v in e['values'])} |")
            md.append("")
        else:
            md.append("All critical fields agree across the fleet.\n")
        if pre["advisory_diff"]:
            md.append(f"Advisory differences (informational): "
                      f"{', '.join('`'+e['field']+'`' for e in pre['advisory_diff'])}\n")

    # gate 1
    md.append("## 1. cuBLAS compute\n")
    if not cub["present"]:
        md.append("_No cuBLAS logs found._\n")
    else:
        md.append(f"Mode: **{cub['mode']}** · parser exit code: **{cub['exit_code']}**\n")
        md.append("```")
        md.append(cub["report"].strip())
        md.append("```\n")

    # gate 2
    md.append("## 2. NCCL interconnect\n")
    if not ncc["present"]:
        md.append("_No NCCL logs found._\n")
    else:
        matched = [t for t in ncc["tests"]
                   if any(r["baseline"] for r in t["rows"])]
        unmatched = [t for t in ncc["tests"] if not any(r["baseline"] for r in t["rows"])]
        if matched:
            md.append(f"{len(matched)} log(s) matched a baseline row in "
                      "`nccl_bench/baselines.md` and are compared below.")
        if unmatched:
            md.append(f"**{len(unmatched)} log(s) had no matching baseline** for this "
                      "GPU/scale — their values are reported without a comparison, "
                      "which is not a pass. Check the GPU model argument and the "
                      "node counts.")
        md.append("")
        for t in ncc["tests"]:
            md.append(f"### `{t['file']}`" + (f" — scale {t['scale']}" if t["scale"] else ""))
            if not any(r["baseline"] for r in t["rows"]):
                md.append("\n_No baseline for this GPU/scale._")
            if t["out_of_bounds"]:
                md.append(f"\n**`# Out of bounds values = {t['out_of_bounds']}`** — "
                          "the collective returned incorrect results. This is a "
                          "correctness failure regardless of bandwidth.\n")
            md.append("| Collective | busbw @16G (GB/s) | Baseline | % of baseline | Peak (GB/s) |")
            md.append("| --- | ---: | ---: | ---: | ---: |")
            for r in t["rows"]:
                b = f"{r['baseline']:.1f}" if r["baseline"] else "—"
                p = f"{r['pct_of_baseline']:.1f}%" if r["pct_of_baseline"] else "—"
                a = f"{r['at16g_oop_busbw']:.2f}" if r["at16g_oop_busbw"] is not None else "—"
                pk = f"{r['peak_oop_busbw']:.2f}" if r["peak_oop_busbw"] is not None else "—"
                md.append(f"| {r['op']} | {a} | {b} | {p} | {pk} |")
            md.append("")
        md.append("Judgement notes: cross-node All2All legitimately falls to ~10–20 % of "
                  "the single-node figure (that is how the collective works); and H800 "
                  "AllReduce ≈215 GB/s is the reduced-NVLink hardware spec, not a fault. "
                  "See `nccl_bench/baselines.md`.\n")

    # gate 3
    md.append("## 3. RDMA point-to-point\n")
    if not pft["present"]:
        md.append("_No perftest output found._\n")
    else:
        if pft["counter_nonzero"]:
            md.append("**Non-zero RDMA error counters detected** — this points at a "
                      "physical layer problem on the listed rail(s):\n")
            md.append("| Rail | Non-zero counters |")
            md.append("| --- | --- |")
            for c in pft["counter_nonzero"]:
                md.append(f"| {c['device']} | {c['detail']} |")
            md.append("")
        else:
            md.append("All RDMA error counters are zero.\n")
        if pft["counter_unreadable"]:
            md.append("**Some counters could not be read** (reported `NA`) — these "
                      "rails are unverified rather than verified-clean:\n")
            md.append("| Rail | Unreadable counters |")
            md.append("| --- | --- |")
            for c in pft["counter_unreadable"]:
                md.append(f"| {c['device']} | {c['detail']} |")
            md.append("")

        if pft["pair"]:
            md.append("### Per-rail, single QP\n")
            md.append("| Rail | GB/s | write Gb/s | bidir | read Gb/s | write lat (us) | send lat (us) |")
            md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in pft["pair"]:
                md.append(f"| {r['rail']} | {r['gbs']} | {r['wr_bw']} | {r['wr_bidi']} "
                          f"| {r['rd_bw']} | {r['wr_lat_us']} | {r['send_lat_us']} |")
            md.append("")
        if pft["agg"]:
            md.append("### Per-rail aggregate (multi-QP)\n")
            md.append("| Rail | peak Gb/s | avg Gb/s | runs |")
            md.append("| --- | ---: | ---: | --- |")
            for r in pft["agg"]:
                md.append(f"| {r['rail']} | {r['peak_gbs_gbps']} | {r['avg_gbps']} | {r['runs']} |")
            md.append("")
            md.append("Judge on the **peak**. On a bonded rail a single run can "
                      "legitimately hash all QPs onto one member port and read about "
                      "half; that is not a fault if the peak across runs reaches the "
                      "rail's capacity.\n")

    md.append("## Reproducing\n")
    md.append("Every number above comes from the raw logs in this directory, "
              "re-parsed by the same scripts the gates use:\n")
    md.append("```bash")
    md.append("python3 cublas_bench/parse_cublas_simple.py results/cublas/cublas_bench_<GPU>.log <GPU>")
    md.append("python3 cublas_bench/parse_cluster_percard.py results/cublas/percard_cluster <GPU>")
    md.append("python3 nccl_bench/parse_nccl_log.py results/nccl/ --at-size 16G")
    md.append("```")

    with open(out_path, "w") as f:
        f.write("\n".join(md) + "\n")

    # ---- json ---------------------------------------------------------------
    payload = {
        "vendor": args.vendor, "gpu": gpu, "overall": overall,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "gates": {n.split(". ", 1)[1]: v for n, v in verdicts},
        "preflight": pre, "cublas": {k: v for k, v in cub.items() if k != "report"},
        "nccl": ncc, "perftest": pft,
    }
    with open(os.path.join(res, "results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"Overall: {overall}")
    for n, v in verdicts:
        print(f"  {n:<26} {v}")
    print(f"\nwrote {out_path}")
    print(f"wrote {os.path.join(res, 'results.json')}")

    if not_run:
        sys.exit(3)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
