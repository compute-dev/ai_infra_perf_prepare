#!/bin/bash
# CUDA runtime environment diagnostic script
# Purpose: when cublasMatmulBench_* reports "libcublasLt.so.12: cannot open shared object file",
#          use this script to quickly locate the problem.
# Usage: bash diagnose_cuda.sh

echo "================ 1. NVIDIA Driver ================"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | head -20
else
    echo "[FAIL] nvidia-smi not found; driver not installed or not in PATH"
fi

echo ""
echo "================ 2. CUDA Toolkit Install Directory ================"
for d in /usr/local/cuda /usr/local/cuda-12* /opt/cuda /opt/cuda-12*; do
    if [[ -d "$d" ]]; then
        echo "[OK] found: $d"
    fi
done
if command -v nvcc >/dev/null 2>&1; then
    echo "[OK] nvcc: $(which nvcc)"
    nvcc --version | tail -1
else
    echo "[WARN] nvcc not found (not necessarily fatal; cuBLAS runtime libs are sufficient)"
fi

echo ""
echo "================ 3. Find libcublasLt.so.12 ================"
FOUND=$(find /usr/local /usr/lib /opt /lib64 -name 'libcublasLt.so*' 2>/dev/null)
if [[ -n "$FOUND" ]]; then
    echo "[OK] libcublasLt versions present on system:"
    echo "$FOUND"
else
    echo "[FAIL] no libcublasLt.so* found on system — need to install CUDA Toolkit (includes cuBLAS)"
fi

echo ""
echo "================ 4. Current LD_LIBRARY_PATH ================"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-(empty)}"

echo ""
echo "================ 5. ldd check: can the binary load all deps? ================"
BIN="./cublasMatmulBench_H20"
if [[ -x "$BIN" ]]; then
    echo "ldd $BIN | grep -E 'cublas|cuda|not found':"
    ldd "$BIN" 2>&1 | grep -E 'cublas|cuda|not found' || echo "(all dependencies found)"
else
    echo "[SKIP] $BIN does not exist or is not executable"
fi

echo ""
echo "================ 6. /etc/ld.so.conf.d/ Configuration ================"
grep -rH 'cuda' /etc/ld.so.conf /etc/ld.so.conf.d/ 2>/dev/null || echo "(no CUDA-related ldconfig entries)"

echo ""
echo "================ Fix Suggestions ================"
cat <<'EOF'
A) CUDA installed but env var missing (most common):
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
   Re-run: GPU_NAME=H20 bash auto_cublas_bench.sh

B) CUDA not installed at all:
   # CentOS / RHEL
   yum install -y cuda-toolkit-12-x   # or matching minor version
   # Ubuntu / Debian
   apt-get install -y cuda-toolkit-12-x

C) Only need cuBLAS runtime (without full toolkit):
   # Download the matching libcublas / libcublasLt packages from NVIDIA

D) Run inside an NVIDIA official container (recommended, avoids host pollution):
   docker run --gpus all -v $(pwd):/work -w /work \
       nvcr.io/nvidia/pytorch:25.10-py3 \
       bash -c "GPU_NAME=H20 bash auto_cublas_bench.sh"
EOF
