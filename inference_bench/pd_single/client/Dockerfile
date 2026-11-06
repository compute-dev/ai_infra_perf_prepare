# pdbench pinned client image: a pinned vLLM release (only its `vllm bench serve` is used) plus the kit.
# It only sends HTTP requests, so the server under test may run on any hardware and any vLLM implementation;
# this image itself needs no GPU.
#
# Build (from inference_bench/pd_single/ — the build context must be that directory):
#   docker build --network=host -f client/Dockerfile --build-arg VLLM_VERSION=0.24.0 -t pdbench-client:0.24.0 .
#   If python:3.12-slim cannot be pulled, any image with python3 (>=3.10) and venv works:
#   --build-arg BASE_IMAGE=<image>
# Run:
#   docker run --rm --net=host -v <model dir>:/model:ro -v <dataset dir>:/data:ro -v <output dir>:/out pdbench-client:0.24.0 \
#       run --role p --base-url http://<server host>:<port> --model-path /model --served-model-name <name> \
#           --dataset-dir /data --out /out --manifest /out/vendor_manifest.yaml --server-log /out/serve_p.log
#
# The version must match PROTO["client_vllm_version"] in kit/pdbench.py, otherwise C13 FAILs.
#
# Known issue: with old Docker (19.03.15 as measured) and this image's default base (Ubuntu 24.04, glibc 2.39),
#   creating a thread inside the container fails (RuntimeError: can't start new thread); adding
#   `--security-opt seccomp=unconfined` restores it. The symptom matches "the old default seccomp profile does
#   not know the syscalls newer glibc uses", but the root cause was not confirmed further. Not seen on newer
#   Docker, and not validated there either.
# Validated scope of this image: it builds on the reference platform (using another Python 3.12 image as the
#   base, because that validation environment had no access to Docker Hub — so the default python:3.12-slim is
#   unvalidated), and inside the container `vllm --version` reports 0.24.0 and `selftest-client` produces
#   request-body hashes matching the golden set.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
ARG VLLM_VERSION
ARG PIP_INDEX_URL
RUN test -n "$VLLM_VERSION" || (echo "--build-arg VLLM_VERSION=<version> is required" && false)
# venv isolation: unaffected by the Python packages the base image ships.
# PIP_CONFIG_FILE=/dev/null shadows the base image's pip.conf. Some base images (NVIDIA NGC ones, for example)
# append an extra-index-url there; when that index is unreachable, pip retries for minutes on every dependency
# (measured: vllm's 200+ dependencies nearly stalled, with 849 DNS retries).
# Note that `pip --isolated` does NOT help here (measured: it still reads /etc/xdg/pip/pip.conf).
# --progress-bar off avoids "can't start new thread" from the progress-bar thread in a restricted build sandbox.
RUN python3 -m venv /opt/venv && PIP_CONFIG_FILE=/dev/null /opt/venv/bin/pip install --no-cache-dir --progress-bar off ${PIP_INDEX_URL:+-i $PIP_INDEX_URL} "vllm==${VLLM_VERSION}" pandas pyyaml
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1
COPY kit /opt/pdbench/kit
ENV PDBENCH_CLIENT_IMAGE=pdbench-client:${VLLM_VERSION}
ENTRYPOINT ["python3", "-s", "/opt/pdbench/kit/pdbench.py"]
