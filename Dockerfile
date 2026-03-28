# ============================================================
# Metrology - Computer Vision Docker Image
# ============================================================
# Base: CUDA 11.8 + Python 3.10
# ============================================================

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Copy requirements first (for layer caching)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install segment-anything and groundingdino from git
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git

WORKDIR /app/GroundingDINO

RUN pip install --no-cache-dir --no-build-isolation -e .

WORKDIR /app

# Copy application code
COPY *.py /app/
COPY config.py /app/

# Create output directory
RUN mkdir -p /app/results_images
RUN mkdir -p /app/weights

# Set entry point
CMD ["python", "main_pipeline.py"]
