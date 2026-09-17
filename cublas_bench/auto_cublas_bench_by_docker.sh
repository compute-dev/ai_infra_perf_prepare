#!/bin/bash
# Run cuBLAS GEMM benchmark inside a Docker container.
#
# The cublasMatmulBench_<GPU> binaries ship in this repo, NOT in the image.
# The image only provides the CUDA runtime libraries.
#
# Usage:
#   GPU_NAME=H800 image=nvcr.io/nvidia/pytorch:25.10-py3 bash auto_cublas_bench_by_docker.sh
#   GPU_NAME=H800 image=<img> DOCKER_TTY=1 bash auto_cublas_bench_by_docker.sh   # interactive TTY
#
# Required:
#   image / IMAGE   Docker image to use (no default; must be set)
#
# Optional:
#   GPU_NAME        GPU model (default: H20)
#   DOCKER_TTY=1    Add -t for interactive TTY (default: off; -i is always used)
#   EXTRA_ARGS      Extra docker run arguments (default: none)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE="${image:-${IMAGE:-}}"
if [[ -z "$IMAGE" ]]; then
    echo "Error: 'image' or 'IMAGE' environment variable is required." >&2
    echo "Usage: GPU_NAME=H800 image=<docker_image> bash $(basename "$0")" >&2
    echo "Note: the image provides the CUDA runtime; the cublasMatmulBench binaries" >&2
    echo "      ship in this repo and are mounted into the container." >&2
    exit 2
fi

GPU_NAME="${GPU_NAME:-H20}"

# -i only (not -it) so the script works in non-interactive shells.
# DOCKER_TTY=1 adds -t when a human is watching an interactive terminal.
TTY_FLAG="-i"
if [[ "${DOCKER_TTY:-0}" == "1" ]]; then
    TTY_FLAG="-it"
fi

docker run --gpus all --shm-size=400gb --rm ${TTY_FLAG} --privileged \
    --entrypoint bash \
    -v "$SCRIPT_DIR":/work -w /work \
    ${EXTRA_ARGS:-} \
    "$IMAGE" \
    -c "GPU_NAME=${GPU_NAME} bash auto_cublas_bench.sh"
