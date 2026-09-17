#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified cuBLAS result parser.

Usage:
  # Method 1: read from a file (specify GPU model)
  python3 parse_cublas_simple.py cublas_bench.log B200

  # Method 2: pipe (live parsing)
  bash cublas_bench.sh 2>&1 | tee cublas_bench.log | python3 parse_cublas_simple.py - B200

  # Method 3: read from stdin
  cat cublas_bench.log | python3 parse_cublas_simple.py - B200

Supported GPU models: GB300, GB200, B300, B200, H100, H200, H800, A800, H20, L20
Baseline data source: cublasMatmulBenchBaseline.yaml
"""

import os
import re
import sys

import yaml

TOLERANCE = 0.05  # 5%

# Baseline YAML file path (same directory as this script)
# Structure: GPU model -> precision -> official TFLOPs
# (explicit key addressing eliminates the row/column misalignment bug of CSV)
BASELINE_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'cublasMatmulBenchBaseline.yaml')


def load_official(gpu_name, yaml_path=BASELINE_YAML):
    """Load official TFLOPs data for the given GPU model from the YAML file.

    Returns dict: {precision: TFLOPs}. Missing/unparseable values are skipped.
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    # Case-insensitive match on the top-level GPU key
    key_map = {str(k).upper(): k for k in data}
    if gpu_name.upper() not in key_map:
        raise ValueError(
            f"GPU '{gpu_name}' not found in baseline YAML. "
            f"Available: {sorted(key_map.keys())}")

    raw = data[key_map[gpu_name.upper()]] or {}
    official = {}
    for precision, value in raw.items():
        if value is None:
            continue
        try:
            official[str(precision).strip()] = float(value)
        except (TypeError, ValueError):
            continue
    return official


def parse_and_compare(content, official):
    """Parse content and compare against baseline.

    Pass/fail semantics (single-sided):
      - Only "measured < official * (1 - TOLERANCE)" is a FAIL
        (performance does not meet the official baseline).
      - Measured at or above baseline is a PASS; readings above tolerance
        are annotated "PASS (high)" for visibility but do not affect exit code.

    Returns:
      0  every evaluated precision met the baseline
      1  at least one evaluated precision was below it
      3  NOTHING was evaluated — the log contained no recognisable result.
         Reported as an error rather than a pass: a crashed run, a truncated
         log, or a wrong GPU name would otherwise silently "pass".
    """
    # Extract Gflops values
    # Key points:
    #   1. Precision keywords use \b word-boundary matching
    #   2. cublasMatmulBench output structure: precision label on one line,
    #      Gflops=xxx several lines later; cross-line matching is required
    #      (re.DOTALL enabled)
    #   3. Negative lookahead (?!...) ensures we walk from the current
    #      precision label to Gflops= without hitting another precision label;
    #      this correctly handles cases like "FP4 has no script\nFP8\n...Gflops=..."
    #      where FP4's match breaks on encountering FP8, letting FP8 match
    #      its own Gflops value.
    precisions = r'FP4|FP8|INT8|FP16|BF16|TF32|FP32'
    pattern = (
        rf'\b({precisions})\b'
        rf'(?:(?!\b(?:{precisions})\b).)*?'
        r'Gflops\s*=\s*([0-9.]+)'
    )
    matches = re.findall(pattern, content, re.DOTALL)

    # Same precision may match multiple times (multiple shapes);
    # keep only the first result for clean display
    seen = set()
    uniq_matches = []
    for precision, gflops in matches:
        if precision in seen:
            continue
        seen.add(precision)
        uniq_matches.append((precision, gflops))

    print("\n" + "=" * 70)
    print(f"{'Prec':<8} {'Measured':<14} {'Official':<14} {'Diff(%)':<12} {'Result'}")
    print("-" * 70)

    all_pass = True
    n_evaluated = 0
    for precision, gflops in uniq_matches:
        measured_tflops = float(gflops) / 1000
        if precision not in official:
            print(f"{precision:<8} {measured_tflops:<14.2f} {'N/A':<14} {'--':>11}  - skip (no baseline)")
            continue
        official_tflops = official[precision]
        diff = ((measured_tflops - official_tflops) / official_tflops) * 100
        n_evaluated += 1

        # Single-sided: only significantly low readings fail; high is not a failure
        if diff < -(TOLERANCE * 100):
            status = "FAIL"
            all_pass = False
        elif diff > (TOLERANCE * 100):
            status = "PASS (high)"
        else:
            status = "PASS"

        print(f"{precision:<8} {measured_tflops:<14.2f} {official_tflops:<14.0f} "
              f"{diff:>+11.2f} {status}")

    print("-" * 70)

    if n_evaluated == 0:
        # Nothing was compared. Do NOT report this as a pass — see the docstring.
        if uniq_matches:
            print("Result: NO BASELINE — parsed "
                  f"{len(uniq_matches)} precision(s), but the baseline file has no "
                  f"entry for any of them.")
            print("        Check that the GPU model argument matches a key in "
                  f"{os.path.basename(BASELINE_YAML)}.")
        else:
            print("Result: NO RESULTS PARSED — the log contains no recognisable "
                  "'<PRECISION> ... Gflops = <value>' output.")
            print("        The benchmark probably did not run (crashed, wrong "
                  "container, or an error before the first timing).")
        print("        This is NOT a pass.")
        print("=" * 70 + "\n")
        return 3

    print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILED'} "
          f"({n_evaluated} precision(s) evaluated; "
          f"criterion: measured >= official x (1 - {TOLERANCE * 100:.0f}%))")
    print("=" * 70 + "\n")

    return 0 if all_pass else 1



def _print_usage():
    print(__doc__)


if __name__ == '__main__':
    # Parse arguments: [log_file] <gpu_name>
    # log_file '-' or omitted means read from stdin
    args = sys.argv[1:]
    if len(args) == 1:
        log_file, gpu_name = '-', args[0]
    elif len(args) >= 2:
        log_file, gpu_name = args[0], args[1]
    else:
        _print_usage()
        sys.exit(2)

    try:
        official = load_official(gpu_name)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    if log_file == '-' or not log_file:
        content = sys.stdin.read()
    else:
        with open(log_file, 'r') as f:
            content = f.read()

    print(f"Baseline GPU: {gpu_name.upper()} (from {os.path.basename(BASELINE_YAML)})")
    rc = parse_and_compare(content, official)
    sys.exit(rc)

