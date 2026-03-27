#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版cuBLAS结果解析工具

# 方法1: 从文件读取
python3 parse_cublas_simple.py cublas_bench.log

# 方法2: 使用管道（实时解析）
bash cublas_bench.sh 2>&1 | tee cublas_bench.log | python3 parse_cublas_simple.py

# 方法3: 从标准输入读取
cat cublas_bench.log | python3 parse_cublas_simple.py

"""

import re
import sys

# B200官方数据 (TFOPs)
OFFICIAL = {'FP4': 6339, 'FP8': 2880, 'FP16': 1437, 'BF16': 1526, 'TF32': 735, 'FP32': 70}
TOLERANCE = 0.05  # 5%


def parse_and_compare(content):
    """解析内容并对比"""
    # 提取Gflops值
    pattern = r'(FP4|FP8|FP16|BF16|TF32|FP32)\s+.*?Gflops\s*=\s*([0-9.]+)'
    matches = re.findall(pattern, content, re.DOTALL)

    print("\n" + "=" * 70)
    print(f"{'精度':<8} {'实测(TFOPs)':<14} {'官方(TFOPs)':<14} {'差异(%)':<12} {'结果'}")
    print("-" * 70)

    all_pass = True
    for precision, gflops in matches:
        measured_tflops = float(gflops) / 1000
        official_tflops = OFFICIAL.get(precision, 0)
        diff = ((measured_tflops - official_tflops) / official_tflops) * 100
        is_pass = abs(diff) <= (TOLERANCE * 100)
        status = "✓ 合格" if is_pass else "✗ 不合格"

        print(f"{precision:<8} {measured_tflops:<14.2f} {official_tflops:<14.0f} "
              f"{diff:>+11.2f} {status}")

        if not is_pass:
            all_pass = False

    print("-" * 70)
    print(f"结果: {'✓ 全部合格' if all_pass else '✗ 存在不合格项'} (允许波动: ±{TOLERANCE * 100}%)")
    print("=" * 70 + "\n")

    return all_pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 从文件读取
        with open(sys.argv[1], 'r') as f:
            content = f.read()
    else:
        # 从标准输入读取
        content = sys.stdin.read()

    all_pass = parse_and_compare(content)
    sys.exit(0 if all_pass else 1)