#!/bin/bash
# CUDA 运行时环境诊断脚本
# 用途: 当 cublasMatmulBench_* 报 "libcublasLt.so.12: cannot open shared object file"
#       时, 用本脚本快速定位问题
# 用法: bash diagnose_cuda.sh

echo "================ 1. NVIDIA 驱动 ================"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | head -20
else
    echo "[FAIL] 未找到 nvidia-smi, 驱动未安装或不在 PATH 中"
fi

echo ""
echo "================ 2. CUDA Toolkit 安装目录 ================"
for d in /usr/local/cuda /usr/local/cuda-12* /opt/cuda /opt/cuda-12*; do
    if [[ -d "$d" ]]; then
        echo "[OK] 发现: $d"
    fi
done
if command -v nvcc >/dev/null 2>&1; then
    echo "[OK] nvcc: $(which nvcc)"
    nvcc --version | tail -1
else
    echo "[WARN] 未找到 nvcc (不一定致命, 只要有 cuBLAS 运行时库即可)"
fi

echo ""
echo "================ 3. 查找 libcublasLt.so.12 ================"
FOUND=$(find /usr/local /usr/lib /opt /lib64 -name 'libcublasLt.so*' 2>/dev/null)
if [[ -n "$FOUND" ]]; then
    echo "[OK] 系统中存在以下版本的 libcublasLt:"
    echo "$FOUND"
else
    echo "[FAIL] 系统中没有 libcublasLt.so* —— 需要安装 CUDA Toolkit (含 cuBLAS)"
fi

echo ""
echo "================ 4. 当前 LD_LIBRARY_PATH ================"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(空)}"

echo ""
echo "================ 5. 用 ldd 看二进制实际能否加载所有依赖 ================"
BIN="./cublasMatmulBench_H20"
if [[ -x "$BIN" ]]; then
    echo "ldd $BIN | grep -E 'cublas|cuda|not found':"
    ldd "$BIN" 2>&1 | grep -E 'cublas|cuda|not found' || echo "(全部依赖均已找到)"
else
    echo "[SKIP] $BIN 不存在或无可执行权限"
fi

echo ""
echo "================ 6. /etc/ld.so.conf.d/ 配置 ================"
grep -rH 'cuda' /etc/ld.so.conf /etc/ld.so.conf.d/ 2>/dev/null || echo "(无 CUDA 相关 ldconfig 条目)"

echo ""
echo "================ 修复建议 ================"
cat <<'EOF'
A) 已装 CUDA 但环境变量缺失 (最常见):
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
   再次执行: GPU_NAME=H20 bash auto_cublas_bench.sh

B) CUDA 完全未装:
   # CentOS / RHEL / TencentOS
   yum install -y cuda-toolkit-12-x   # 或对应小版本
   # Ubuntu / Debian
   apt-get install -y cuda-toolkit-12-x

C) 不想装完整 Toolkit, 只装 cuBLAS 运行时:
   # 从 NVIDIA 官网下载对应版本的 libcublas / libcublasLt 包

D) 在 NVIDIA 官方容器内运行 (推荐, 避免污染主机):
   docker run --gpus all -v $(pwd):/work -w /work \
       nvcr.io/nvidia/pytorch:25.10-py3 \
       bash -c "GPU_NAME=H20 bash auto_cublas_bench.sh"
EOF