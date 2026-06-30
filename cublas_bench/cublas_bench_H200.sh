chmod +x cublasMatmulBench_H200_H100

echo "FP8 All Zeros, H200 GEMM TFLOPS：1505, H100 GEMM TFLOPS：1101"
./cublasMatmulBench_H200_H100 -P=qqssq -m=4224 -n=2048 -k=16384 -T=1000 -ta=1 -B=0

echo "INT8 All Zeros, H200 GEMM TFLOPS：1522, H100 GEMM TFLOPS：1059"
./cublasMatmulBench_H200_H100 -P=bisb_imma -m=8192 -n=4224 -k=16384 -T=1000 -ta=1 -B=0

echo "FP16 All Zeros, H200 GEMM TFLOPS：749, H100 GEMM TFLOPS：540"
./cublasMatmulBench_H200_H100 -P=hsh -m=12288 -n=9216 -k=32768 -T=1000 -tb=1 -B=0

echo "BF16 All Zeros, H200 GEMM TFLOPS：759, H100 GEMM TFLOPS：531"
./cublasMatmulBench_H200_H100 -P=tst -m=12288 -n=9216 -k=32768 -T=1000 -tb=1 -B=0

echo "TF32 All Zeros, H200 GEMM TFLOPS：447, H100 GEMM TFLOPS：322"
./cublasMatmulBench_H200_H100 -P=sss_fast_tf32 -m=8192 -n=4224 -k=16384 -T=1000 -tb=1 -B=0

echo "FP64 All Zeros, H200 GEMM TFLOPS：66, H100 GEMM TFLOPS：56"
./cublasMatmulBench_H200_H100 -P=ddd -m=4224 -n=2048 -k=16384 -T=1000 -tb=1 -B=0

echo "FP32 All Zeros, H200 GEMM TFLOPS：55, H100 GEMM TFLOPS：49"
./cublasMatmulBench_H200_H100 -P=sss -m=4224 -n=2048 -k=16384 -T=1000 -tb=1 -B=0