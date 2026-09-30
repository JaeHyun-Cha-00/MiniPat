# Training + inference environment for MiniPat (RunPod or any host with an NVIDIA driver for CUDA 12.6+).
#
#   docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t minipat .
#   docker run --gpus all -e HF_TOKEN -v /workspace:/workspace minipat \
#       python training/train_lora.py --model qwen35_4b    # -> /workspace/checkpoints/qwen35_4b
#
# On RunPod the network volume is mounted at /workspace; the model cache and checkpoints go
# there so they survive pod restarts. Data comes from the HF dataset (see data_pipeline/push_dataset.py).

# devel (not runtime) image: causal-conv1d below is compiled with nvcc.
FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf_cache

WORKDIR /app

COPY requirements/train.txt requirements/eval.txt requirements/
RUN pip install -r requirements/train.txt -r requirements/eval.txt
# score.py (COMET) needs its own environment; see requirements/score.txt.

# Source-only package. Compiling for every GPU arch is slow, so build for the usual RunPod cards:
# 8.0 A100, 8.6 A6000/3090, 8.9 L40S/4090, 9.0 H100. Override with --build-arg if needed.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS=4 \
    pip install --no-build-isolation causal-conv1d==1.7.0

COPY training/ training/
COPY evaluation/ evaluation/

# Recorded in each run's run_info.json, since .git is not copied into the image.
ARG GIT_SHA=unknown
ENV MINIPAT_GIT_SHA=${GIT_SHA}
