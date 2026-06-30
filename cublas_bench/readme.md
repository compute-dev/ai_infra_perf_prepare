```shell
# B 和 GB 用cuda13以上的镜像，pytorch2602
# 非B 和 GB 用cuda12的镜像，pytorch2505，nvcr.io/nvidia/pytorch:25.05-py3
docker pull rozinnn/ai_infra_bench:nccltest_x86_nccl2.26.5-cuda12.9-H

cd /path/to/cublas_bench
image="rozinnn/ai_infra_bench:nccltest_x86_nccl2.26.5-cuda12.9-H"
docker run --gpus all --shm-size=400gb --rm -it --privileged\
    -v $(pwd):/work -w /work \
    ${image} \
    bash -c "GPU_NAME=H20 bash auto_cublas_bench.sh"

```