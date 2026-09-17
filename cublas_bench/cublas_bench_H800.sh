chmod +x cublasMatmulBench_H800

#echo "FP8"
#./cublasMatmulBench_H800 -P=qqssq -m=4224 -n=2048 -k=16384 -T=1000 -ta=1 -B=0

echo "INT8 ta=1"
./cublasMatmulBench_H800 -P=bisb_imma -m=8192 -n=4224 -k=16384 -T=1000 -ta=1 -B=0

echo "INT8 ta=0"
./cublasMatmulBench_H800 -P=bisb_imma -m=8192 -n=4224 -k=16384 -T=1000 -ta=0 -B=0

echo "FP16"
./cublasMatmulBench_H800 -P=hsh -m=12288 -n=9216 -k=32768 -T=1000 -tb=1 -B=0

echo "TF32"
./cublasMatmulBench_H800 -P=sss_fast_tf32 -m=8192 -n=4224 -k=16384 -T=1000 -ta=1 -B=0

echo "FP32"
./cublasMatmulBench_H800 -P=sss -m=4224 -n=2048 -k=16384 -T=1000 -tb=1 -B=0