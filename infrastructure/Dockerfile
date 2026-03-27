FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install PyTorch with CUDA 12.8
RUN pip install --no-cache-dir \
    torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Install training dependencies
COPY infrastructure/requirements-train.txt /tmp/requirements-train.txt
RUN pip install --no-cache-dir -r /tmp/requirements-train.txt

# SageMaker expects code at /opt/ml/code
WORKDIR /opt/ml/code

# Copy SageMaker wrapper scripts (from project root context)
COPY sagemaker/entry_point.py /opt/ml/code/entry_point.py
COPY sagemaker/train_wrapper.py /opt/ml/code/train_wrapper.py

# SageMaker entry point
ENV SAGEMAKER_PROGRAM=entry_point.py
