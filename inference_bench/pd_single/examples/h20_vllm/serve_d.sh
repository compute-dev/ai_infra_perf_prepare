# Reference adaptation (NVIDIA H20 + vLLM). Items marked "site-specific" belong to the reference
# environment, not to the protocol — replace them with what your hardware needs.
# Starts the D endpoint on port 8020.
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=<allreduce backend>   # site-specific
export NCCL_SOCKET_IFNAME=<NIC name>                           # site-specific
export GLOO_SOCKET_IFNAME=<NIC name>                           # site-specific
rm -rf /root/.cache/vllm/torch_compile_cache
vllm serve <model dir> --host 0.0.0.0 --port 8020 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --tool-call-parser hy_v3 \
  --enable-auto-tool-choice \
  --served-model-name pdbench-model \
  --reasoning-parser hy_v3 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching
