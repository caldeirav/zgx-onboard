# =============================================================================
# ZGX Nano CUDA 13 Dev Container for Blackwell (RTX 50 series)
# =============================================================================
# Target: CUDA 13 Development environment for Grace-Blackwell / Arm64
# Purpose: General-purpose container for CUDA demos and experiments
#
# Build:  docker build -t zgx-cuda docker/
# Run:    docker run -d --gpus all --shm-size=100g \
#             -v $(pwd):/workspace --name zgx_cuda zgx-cuda
#
# Note: Uses NVIDIA PyTorch image with CUDA 13.0 for full Blackwell support
# See: https://build.nvidia.com/spark/unsloth/instructions
# =============================================================================

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# Install System Tools
# - git, procps: Required for Cursor/VSCode to attach correctly
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    htop \
    nano \
    procps \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Install Python Dependencies
# - PyTorch is already included in the base image with CUDA 13.0
# - Install additional ML/AI packages
# Note: Using --no-deps for unsloth to avoid dependency conflicts with base image
# See: https://build.nvidia.com/spark/unsloth/instructions
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    transformers \
    peft \
    hf_transfer \
    "datasets==4.3.0" \
    "trl==0.26.1"

# Install unsloth without deps (as per NVIDIA instructions for Blackwell)
RUN pip install --no-deps \
    unsloth \
    unsloth_zoo \
    bitsandbytes

# -----------------------------------------------------------------------------
# Install Additional Dependencies for AML Investigation Agent
# - mlflow: Experiment tracking and agent tracing
# - networkx: Financial graph database simulation
# - kaggle: Dataset download from Kaggle
# - polars: Fast DataFrame library for large dataset processing
# - google-genai: Gemini API for LLM-as-judge evaluation
# - langgraph: Agent framework with state management
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    mlflow \
    networkx \
    kaggle \
    polars \
    google-genai \
    safetensors \
    tokenizers \
    scikit-learn \
    langgraph \
    langchain-core \
    langchain-ollama \
    accelerate \
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
