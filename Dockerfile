# syntax=docker/dockerfile:1.7

# ROS 2 Humble targets Ubuntu 22.04, while NVIDIA's CUDA 11.3 image targets
# Ubuntu 20.04. Importing the toolkit into Jammy lets MinkowskiEngine compile
# against the exact CUDA minor version used by the PyTorch cu113 wheel.
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 AS cuda

FROM ros:humble-ros-base-jammy

ARG DEBIAN_FRONTEND=noninteractive
# This post-v0.5.4 commit still reports 0.5.4 and adds Python 3.10/PyTorch 1.11+
# compatibility. Keep it aligned with deploy/setup_venv.sh.
ARG MINKOWSKI_ENGINE_COMMIT=02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
ARG MINKOWSKI_ENGINE_SHA256=9ac2730bff659202400a76abf370e9690caa1b68edd430629422192fccd7af02
ARG TORCH_CUDA_ARCH_LIST="8.6+PTX"
ARG CHECKPOINT_URL="https://github.com/subhransu10/sparseunet4d/releases/latest/download/best.pt"
ARG CHECKPOINT_SHA256="65f7525f00a4a490df30ec91b5db713d865f30dffd76b4c7f9dfcbc353e31f1c"

LABEL org.opencontainers.image.source="https://github.com/subhransu10/sparseunet4d" \
      org.opencontainers.image.description="SparseUNet4D ROS 2 Humble inference image"

COPY --from=cuda /usr/local/cuda-11.3 /usr/local/cuda-11.3

ENV CUDA_HOME=/usr/local/cuda-11.3 \
    PATH=/usr/local/cuda-11.3/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:${LD_LIBRARY_PATH} \
    SU4D_BACKEND=me \
    PYTHONPATH=/opt/sparseunet4d \
    OMP_NUM_THREADS=6 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN ln -s /usr/local/cuda-11.3 /usr/local/cuda \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        g++-10 \
        gcc-10 \
        libopenblas-dev \
        python3-dev \
        python3-pip \
        ros-humble-nav-msgs \
        ros-humble-sensor-msgs-py \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade "pip<25" setuptools==59.6.0 wheel \
    && python3 -m pip install \
        torch==1.12.1+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
    && python3 -m pip install \
        ninja \
        numpy==1.26.4 \
        pyyaml==6.0.3 \
        scipy==1.15.3

RUN curl --fail --location --retry 5 \
        "https://github.com/NVIDIA/MinkowskiEngine/archive/${MINKOWSKI_ENGINE_COMMIT}.tar.gz" \
        --output /tmp/minkowski-engine.tar.gz \
    && echo "${MINKOWSKI_ENGINE_SHA256}  /tmp/minkowski-engine.tar.gz" \
        | sha256sum --check --strict \
    && tar -xzf /tmp/minkowski-engine.tar.gz -C /tmp \
    && cd "/tmp/MinkowskiEngine-${MINKOWSKI_ENGINE_COMMIT}" \
    && CC=gcc-10 CXX=g++-10 MAX_JOBS=1 \
       TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
       python3 setup.py install --blas=openblas --force_cuda \
    && cd / \
    && rm -rf /tmp/minkowski-engine.tar.gz \
              "/tmp/MinkowskiEngine-${MINKOWSKI_ENGINE_COMMIT}"

WORKDIR /opt/sparseunet4d

RUN mkdir -p checkpoints/sparseunet4d_semantickitti \
    && curl --fail --location --retry 5 \
        "${CHECKPOINT_URL}" \
        --output checkpoints/sparseunet4d_semantickitti/best.pt \
    && echo "${CHECKPOINT_SHA256}  checkpoints/sparseunet4d_semantickitti/best.pt" \
        | sha256sum --check --strict

COPY . /opt/sparseunet4d

RUN python3 - <<'PY'
import MinkowskiEngine as ME
import rclpy
import sensor_msgs_py
import torch
from sparseunet4d.models.backend import backend

assert torch.__version__ == "1.12.1+cu113", torch.__version__
assert torch.version.cuda == "11.3", torch.version.cuda
assert ME.__version__ == "0.5.4", ME.__version__
assert backend() == "me", backend()
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda)
print("MinkowskiEngine", ME.__version__)
PY

CMD ["python3", "/opt/sparseunet4d/mos_node.py", "--ros-args", \
     "-p", "config:=/opt/sparseunet4d/configs/pretrained_semantickitti.yaml", \
     "-p", "ckpt:=/opt/sparseunet4d/checkpoints/sparseunet4d_semantickitti/best.pt", \
     "-p", "device:=cuda", "-p", "propagate:=false", "-p", "pipeline:=true", \
     "-r", "/sparseunet4d_mos/points:=/velodyne_points", \
     "-r", "/sparseunet4d_mos/odom:=/odom"]
