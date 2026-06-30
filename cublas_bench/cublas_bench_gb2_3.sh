chmod +x cublasMatmulBench_gb2_3
# FP4
echo "FP4"
./cublasMatmulBench_gb2_3 -P=nvoohso -m=9728 -n=16384 -k=8192 -ta=1 -tb=0 -A=1 -B=0 -T=1000 -W=10000 -p=t -sf_p=u

# FP8
echo "FP8"
./cublasMatmulBench_gb2_3 -P=qqssq -m=9728 -n=2048 -k=32768 -ta=1 -tb=0 -A=1 -B=0 -T=1000 -W=10000 -p=t

# FP16
echo "FP16"
./cublasMatmulBench_gb2_3 -P=hsh -m=8192 -n=9728 -k=16384 -ta=0 -tb=1 -A=1 -B=0 -T=1000 -W=10000 -p=t

# BF16
echo "BF16"
./cublasMatmulBench_gb2_3 -P=tst -m=8192 -n=9728 -k=16384 -ta=0 -tb=1 -A=1 -B=0 -T=1000 -W=10000 -p=t

# TF32
echo "TF32"
./cublasMatmulBench_gb2_3 -P=sss_fast_tf32 -m=8192 -n=9728 -k=16384 -ta=0 -tb=1 -A=1 -B=0 -T=1000 -W=10000 -p=t

# FP32
echo "FP32"
./cublasMatmulBench_gb2_3 -P=sss -m=8192 -n=9728 -k=16384 -ta=0 -tb=1 -A=1 -B=0 -T=1000 -W=1000 -p=t
