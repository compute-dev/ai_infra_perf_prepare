chmod +x cublasMatmulBench_H20

echo "FP8 All Zeros, H20 GEMM TFLOPS: 271"
./cublasMatmulBench_H20 -P=qqssq -m=4992 -n=1024 -k=8192 -T=1000 -ta=1 -B=0

echo "INT8 All Zeros, H20 GEMM TFLOPS: 283"
./cublasMatmulBench_H20 -P=bisb_imma -m=8192 -n=2496 -k=16384 -T=1000 -ta=1 -B=0

echo "FP16 All Zeros, H20 GEMM TFLOPS: 141"
./cublasMatmulBench_H20 -P=hsh -m=12288 -n=9216 -k=32768 -T=1000 -tb=1 -B=0

echo "BF16 All Zeros, H20 GEMM TFLOPS: 141"
./cublasMatmulBench_H20 -P=tst -m=12288 -n=9216 -k=32768 -T=1000 -tb=1 -B=0

echo "TF32 All Zeros, H20 GEMM TFLOPS: 69"
./cublasMatmulBench_H20 -P=sss_fast_tf32 -m=8192 -n=4224 -k=16384 -T=1000 -tb=1 -B=0

echo "FP32 All Zeros, H20 GEMM TFLOPS: 31"
./cublasMatmulBench_H20 -P=sss -m=2496 -n=65536 -k=16384 -T=1000 -tb=1 -B=0