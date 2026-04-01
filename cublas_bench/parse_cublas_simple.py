#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified cuBLAS Result Parsing Tool

# Method 1: Read from file
python3 parse_cublas_simple.py cublas_bench.log

# Method 2: Use pipe (real-time parsing)
bash cublas_bench.sh 2>&1 | tee cublas_bench.log | python3 parse_cublas_simple.py

# Method 3: Read from stdin
cat cublas_bench.log | python3 parse_cublas_simple.py

"""

import re
import sys

# B200 official data (TFOPs)
OFFICIAL = {'FP4': 6339, 'FP8': 2880, 'FP16': 1437, 'BF16': 1526, 'TF32': 735, 'FP32': 70}
TOLERANCE = 0.05  # 5%


def parse_and_compare(content):
    """Parse content and compare"""
    # Extract Gflops values
    pattern = r'(FP4|FP8|FP16|BF16|TF32|FP32)\s+.*?Gflops\s*=\s*([0-9.]+)'
    matches = re.findall(pattern, content, re.DOTALL)

    print("\n" + "=" * 70)
    print(f"{'Precision':<10} {'Measured(TFOPs)':<16} {'Official(TFOPs)':<16} {'Diff(%)':<12} {'Result'}")
    print("-" * 70)

    all_pass = True
    for precision, gflops in matches:
        measured_tflops = float(gflops) / 1000
        official_tflops = OFFICIAL.get(precision, 0)
        diff = ((measured_tflops - official_tflops) / official_tflops) * 100
        is_pass = abs(diff) <= (TOLERANCE * 100)
        status = "✓ PASS" if is_pass else "✗ FAIL"

        print(f"{precision:<10} {measured_tflops:<16.2f} {official_tflops:<16.0f} "
              f"{diff:>+11.2f} {status}")

        if not is_pass:
            all_pass = False

    print("-" * 70)
    print(f"Result: {'✓ ALL PASSED' if all_pass else '✗ SOME TESTS FAILED'} (Tolerance: ±{TOLERANCE * 100}%)")
    print("=" * 70 + "\n")

    return all_pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r') as f:
            content = f.read()
    else:
        # Read from stdin
        content = sys.stdin.read()

    all_pass = parse_and_compare(content)
    sys.exit(0 if all_pass else 1)