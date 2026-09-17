chmod +x cublasMatmulBench_B300
# FP4
echo "FP4 暂时没有对应脚本"

# FP8
echo "FP8"
./cublasMatmulBench_B300 -P=qqssq -m=8192 -n=9472 -k=16384 -ta=1 -tb=0 -A=1 -B=0 -T=1000 -W=10000 -p=t

# FP16
echo "FP16"
./cublasMatmulBench_B300 -P=hsh -m=8192 -n=9472 -k=16384 -ta=0 -tb=1 -A=1 -B=0 -T=1000 -W=10000 -p=t

# BF16
echo "BF16"
./cublasMatmulBench_B300 -P=tst -m=8192 -n=9472 -k=16384 -ta=0 -tb=1 -A=1 -B=0 -T=1000 -W=10000 -p=t

# TF32
echo "TF32"
./cublasMatmulBench_B300 -P=sss_fast_tf32 -m=8192 -n=9472 -k=16384 -ta=0 -tb=1 -A=1 -B=0 -T=1000 -W=10000 -p=t

# FP32
echo "FP32"
./cublasMatmulBench_B300 -P=sss -m=8192 -n=9472 -k=16384 -ta=0 -tb=1 -A=1 -B=0 -T=1000 -W=10000 -p=t
