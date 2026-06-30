#!/bin/bash
# cuBLAS GEMM 自动基准测试入口
#   - 根据环境变量 GPU_NAME 选择对应的测试脚本
#   - 测试输出同时写入 cublas_bench_<GPU>.log
#   - 解析脚本将实测 TFLOPs 与 cublasMatmulBenchBaseline.csv 中的官方基线对比
#
# 用法:
#   GPU_NAME=B200 bash auto_cublas_bench.sh
#
# 支持的 GPU_NAME (大小写不敏感):
#   GB300, GB200, B300, B200, H200, H100, H800, A800, H20, L20

set -eo pipefail

# 切到脚本所在目录，保证相对路径可用（无论从哪儿调用都行）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 1. 校验 GPU_NAME ----------
if [[ -z "${GPU_NAME:-}" ]]; then
    echo "错误: 未设置环境变量 GPU_NAME" >&2
    echo "用法: GPU_NAME=<型号> bash $(basename "$0")" >&2
    echo "支持型号: GB300 GB200 B300 B200 H200 H100 H800 A800 H20 L20" >&2
    exit 2
fi

# 统一转大写，方便匹配
GPU_UPPER="$(echo "$GPU_NAME" | tr '[:lower:]' '[:upper:]')"

# ---------- 2. 选择测试脚本与基线型号 ----------
# BENCH_SCRIPT : 实际执行的 cublas_bench_*.sh
# BASELINE_GPU : 传给 parse_cublas_simple.py 的 GPU 型号（用于查 CSV 基线列）
case "$GPU_UPPER" in
    B200)
        BENCH_SCRIPT="cublas_bench_B200.sh";      BASELINE_GPU="B200"  ;;
    B300)
        BENCH_SCRIPT="cublas_bench_B300.sh";      BASELINE_GPU="B300"  ;;
    GB200)
        BENCH_SCRIPT="cublas_bench_GB200.sh";     BASELINE_GPU="GB200" ;;
    GB300)
        BENCH_SCRIPT="cublas_bench_GB300.sh";     BASELINE_GPU="GB300" ;;
    H100)
        BENCH_SCRIPT="cublas_bench_H100.sh"; BASELINE_GPU="H100"  ;;
    H200)
        BENCH_SCRIPT="cublas_bench_H200.sh"; BASELINE_GPU="H200"  ;;
    H800)
        BENCH_SCRIPT="cublas_bench_H800.sh";     BASELINE_GPU="H800"  ;;
    A800)
        BENCH_SCRIPT="cublas_bench_A800.sh";     BASELINE_GPU="A800"  ;;
    H20)
        BENCH_SCRIPT="cublas_bench_H20.sh";       BASELINE_GPU="H20"   ;;
    L20)
        BENCH_SCRIPT="cublas_bench_L20.sh";       BASELINE_GPU="L20"   ;;
    *)
        echo "错误: 不支持的 GPU_NAME='$GPU_NAME'" >&2
        echo "支持型号: GB300 GB200 B300 B200 H200 H100 H800 A800 H20 L20" >&2
        exit 2
        ;;
esac

# ---------- 3. 文件存在性预检查 ----------
if [[ ! -f "$BENCH_SCRIPT" ]]; then
    echo "错误: 测试脚本不存在: $BENCH_SCRIPT" >&2
    exit 2
fi
if [[ ! -f "cublasMatmulBenchBaseline.csv" ]]; then
    echo "错误: 基线文件不存在: cublasMatmulBenchBaseline.csv" >&2
    exit 2
fi
if [[ ! -f "parse_cublas_simple.py" ]]; then
    echo "错误: 解析脚本不存在: parse_cublas_simple.py" >&2
    exit 2
fi

LOG_FILE="cublas_bench_${GPU_UPPER}.log"

echo "========================================================"
echo "  cuBLAS GEMM Benchmark"
echo "  GPU_NAME      : $GPU_UPPER"
echo "  测试脚本      : $BENCH_SCRIPT"
echo "  基线对比型号  : $BASELINE_GPU (cublasMatmulBenchBaseline.csv)"
echo "  日志文件      : $LOG_FILE"
echo "========================================================"

# ---------- 4. 运行前自检：cublasMatmulBench_<XXX> 二进制是否能加载所有动态库 ----------
# 从 BENCH_SCRIPT 中识别它实际调用的二进制 (形如 ./cublasMatmulBench_H20)
BENCH_BIN=$(grep -oE '\./cublasMatmulBench_[A-Za-z0-9_]+' "$BENCH_SCRIPT" | head -1)
if [[ -n "$BENCH_BIN" ]]; then
    BENCH_BIN_PATH="${BENCH_BIN#./}"
    if [[ ! -f "$BENCH_BIN_PATH" ]]; then
        echo "错误: 测试二进制不存在: $BENCH_BIN_PATH" >&2
        exit 2
    fi
    chmod +x "$BENCH_BIN_PATH" 2>/dev/null || true
    if command -v ldd >/dev/null 2>&1; then
        MISSING=$(ldd "$BENCH_BIN_PATH" 2>&1 | grep 'not found' || true)
        if [[ -n "$MISSING" ]]; then
            echo "错误: $BENCH_BIN_PATH 缺失共享库依赖:" >&2
            echo "$MISSING" >&2
            echo "" >&2
            echo "提示: 通常是 CUDA 运行时未配置, 可尝试:" >&2
            echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}" >&2
            echo "或使用 NVIDIA 官方容器 (nvcr.io/nvidia/pytorch:25.10-py3) 运行本脚本" >&2
            echo "也可执行 'bash diagnose_cuda.sh' 做更详细的诊断" >&2
            exit 3
        fi
    fi
fi

# ---------- 5. 执行测试 + 实时解析 ----------
# 用 PIPESTATUS 拿到管道各段的真实退出码
# 注意: PIPESTATUS 在下一条命令执行后会被刷新, 所以必须立刻整体拷贝出来
bash "$BENCH_SCRIPT" 2>&1 | tee "$LOG_FILE" | python3 parse_cublas_simple.py - "$BASELINE_GPU"
RC_ARRAY=("${PIPESTATUS[@]}")
BENCH_RC=${RC_ARRAY[0]:-0}
PARSE_RC=${RC_ARRAY[2]:-0}

# 兜底: 即便上面的 ldd 自检没拦住, 这里再扫一次日志里是否有 "shared libraries" 错误
if grep -q 'error while loading shared libraries' "$LOG_FILE" 2>/dev/null; then
    echo "" >&2
    echo "错误: 测试日志中检测到 '动态库加载失败', 测试未真正执行" >&2
    echo "  详见 $LOG_FILE 中的 'error while loading shared libraries' 行" >&2
    echo "  可执行 'bash diagnose_cuda.sh' 做诊断" >&2
    exit 3
fi

if [[ $BENCH_RC -ne 0 ]]; then
    echo "错误: 测试脚本 $BENCH_SCRIPT 执行失败 (rc=$BENCH_RC)" >&2
    exit "$BENCH_RC"
fi

exit "$PARSE_RC"
