# =============================================================================
# ZGX Nano CUDA Dev Container
# =============================================================================
# Target: CUDA Development environment for Grace-Blackwell / Arm64
# Purpose: General-purpose container for CUDA demos and experiments
#
# Build:  docker build -t zgx-cuda docker/
# Run:    docker run -d --gpus all --shm-size=100g \
#             -v $(pwd):/workspace --name zgx_cuda zgx-cuda
#
# Note: Uses CUDA 12.8 for Blackwell (GB10, sm_120/sm_121) GPU support
# =============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Install System Tools
# - git, procps: Required for Cursor/VSCode to attach correctly
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    htop \
    nano \
    procps \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Install Python Dependencies
# - PyTorch 2.7+ with CUDA 12.8 for Blackwell (GB10) GPU support
# - Transformers ecosystem for LLM inference
# - ipykernel: Required for Cursor notebook support
# - nvitop: Real-time GPU monitoring
# Note: --break-system-packages is safe in containers (isolated environment)
# -----------------------------------------------------------------------------
RUN pip3 install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip3 install --no-cache-dir --break-system-packages \
    transformers \
    accelerate \
    bitsandbytes \
    nvitop \
    huggingface_hub \
    matplotlib \
    pandas \
    numpy \
    seaborn \
    python-dotenv \
    ipykernel \
    ipywidgets \
    psutil

# -----------------------------------------------------------------------------
# Working Directory
# -----------------------------------------------------------------------------
WORKDIR /workspace

# -----------------------------------------------------------------------------
# Default Command: Keep container alive for Cursor to attach
# -----------------------------------------------------------------------------
CMD ["sleep", "infinity"]

