chmod +x cublasMatmulBench_L20

echo "FP8 All Zeros, L20 GEMM TFLOPS: 228"
./cublasMatmulBench_L20 -P=qqssq -m=15360 -n=18176 -k=8192 -T=1000 -ta=1 -B=0 -p=0

echo "INT8 All Zeros, L20 GEMM TFLOPS: 232"
./cublasMatmulBench_L20 -P=bisb_imma -m=40960 -n=52548 -k=16384 -T=1000 -ta=1 -B=0 -p=0

echo "FP16 All Zeros, L20 GEMM TFLOPS: 116"
./cublasMatmulBench_L20 -P=hsh -m=15360 -n=18176 -k=8192 -T=1000 -tb=1 -B=0 -p=0

echo "BF16 All Zeros, L20 GEMM TFLOPS: 114"
./cublasMatmulBench_L20 -P=hsh -m=15360 -n=18176 -k=16384 -T=1000 -tb=1 -B=0 -p=0

echo "TF32 All Zeros, L20 GEMM TFLOPS: 58"
./cublasMatmulBench_L20 -P=sss_fast_tf32 -m=15360 -n=18176 -k=4096 -T=1000 -tb=1 -B=0 -p=0

echo "FP32 All Zeros, L20 GEMM TFLOPS: 48"
./cublasMatmulBench_L20 -P=sss -m=15360 -n=18176 -k=4096 -T=1000 -tb=1 -B=0 -p=0