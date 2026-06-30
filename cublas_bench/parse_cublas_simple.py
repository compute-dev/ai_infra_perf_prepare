#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版cuBLAS结果解析工具

# 方法1: 从文件读取（指定 GPU 型号）
python3 parse_cublas_simple.py cublas_bench.log B200

# ⭐️方法2: 使用管道（实时解析）
bash cublas_bench.sh 2>&1 | tee cublas_bench.log | python3 parse_cublas_simple.py - B200

# 方法3: 从标准输入读取
cat cublas_bench.log | python3 parse_cublas_simple.py - B200

支持的 GPU 型号: GB300, GB200, B300, B200, H100, H200, H800, A800, H20, L20
基线数据来源: cublasMatmulBenchBaseline.csv
"""

import csv
import os
import re
import sys

TOLERANCE = 0.05  # 5%

# 基线 CSV 文件路径（与本脚本同目录）
BASELINE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'cublasMatmulBenchBaseline.csv')

# GPU 简称 -> CSV 列名前缀的映射（用于模糊匹配 CSV 表头）
# 注意：H20/L20 与 H200/L20-xxx 的前缀冲突需要精确匹配（见 load_official）
GPU_ALIASES = {
    'GB300': 'GB300',
    'GB200': 'GB200',
    'B300':  'B300',
    'B200':  'B200',
    'H200':  'H200',
    'H100':  'H100',
    'H800':  'H800',
    'A800':  'A800',
    'H20':   'H20',
    'L20':   'L20',
}


def load_official(gpu_name, csv_path=BASELINE_CSV):
    """从 CSV 加载指定 GPU 型号的官方 TFLOPs 数据。

    返回 dict: {精度: TFLOPs}，缺失值（'—' 或空）会被跳过。
    """
    if gpu_name.upper() not in GPU_ALIASES:
        raise ValueError(f"不支持的 GPU 型号: {gpu_name}，可选: {list(GPU_ALIASES.keys())}")
    key = GPU_ALIASES[gpu_name.upper()]

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        # 找到匹配的列索引：要求 GPU 简称作为独立 token 出现在列名开头，
        # 后面紧跟空格、'-' 或字符串结束，避免 H20 误匹配 "H200-141GBGEMM TFLOPs"
        col_pattern = re.compile(rf'^{re.escape(key)}(?=$|[\s\-])', re.IGNORECASE)
        col_idx = None
        for i, col in enumerate(header):
            if col_pattern.match(col.strip()):
                col_idx = i
                break
        if col_idx is None:
            raise ValueError(f"CSV 中未找到 {gpu_name} 对应的列，表头: {header}")

        official = {}
        for row in reader:
            if not row or not row[0].strip():
                continue
            precision = row[0].strip()
            # 防止该行列数不足
            value = row[col_idx].strip() if col_idx < len(row) else ''
            if not value or value in ('—', '-', 'N/A', 'NA'):
                continue
            try:
                official[precision] = float(value)
            except ValueError:
                continue
        return official


def parse_and_compare(content, official):
    """解析内容并对比"""
    # 提取Gflops值
    pattern = r'(FP4|FP8|INT8|FP16|BF16|TF32|FP32)\s+.*?Gflops\s*=\s*([0-9.]+)'
    matches = re.findall(pattern, content, re.DOTALL)

    print("\n" + "=" * 70)
    print(f"{'精度':<8} {'实测(TFOPs)':<14} {'官方(TFOPs)':<14} {'差异(%)':<12} {'结果'}")
    print("-" * 70)

    all_pass = True
    for precision, gflops in matches:
        measured_tflops = float(gflops) / 1000
        if precision not in official:
            print(f"{precision:<8} {measured_tflops:<14.2f} {'N/A':<14} {'--':>11}  - 跳过（无基线）")
            continue
        official_tflops = official[precision]
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


def _print_usage():
    print(__doc__)


if __name__ == '__main__':
    # 解析参数：[log_file] <gpu_name>
    # log_file 为 '-' 或省略时表示从标准输入读取
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
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(2)

    if log_file == '-' or not log_file:
        content = sys.stdin.read()
    else:
        with open(log_file, 'r') as f:
            content = f.read()

    print(f"基线 GPU: {gpu_name.upper()}（来自 {os.path.basename(BASELINE_CSV)}）")
    all_pass = parse_and_compare(content, official)
    sys.exit(0 if all_pass else 1)