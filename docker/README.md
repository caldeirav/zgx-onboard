# ZGX Nano CUDA Dev Container

This directory contains the Docker configuration for running CUDA demos as a Dev Container.

## Overview

The container provides a complete CUDA development environment optimized for Grace-Blackwell architecture, suitable for:

- **Unified Memory demos** - Testing memory oversubscription and C2C coherency
- **LLM inference benchmarks** - Running large language models with PyTorch
- **CUDA profiling** - Using Nsight Systems for performance analysis
- **General CUDA 13 development** - Any workload requiring CUDA 13.0

### Key Features

- **CUDA 12.6** development environment (Ubuntu 24.04)
- Pre-installed ML stack (PyTorch, Transformers, Accelerate, BitsAndBytes)
- **ipykernel** for Cursor notebook support
- **nvitop** for real-time GPU monitoring

## Quick Start

### 1. Create Environment File

```bash
# From project root
echo "HF_TOKEN=hf_your_huggingface_token_here" > .env
```

### 2. Build and Run

```bash
# Using the helper script
cd docker
./build-and-run.sh

# Or manually from project root
docker build -t zgx-cuda docker/
docker run -d \
    --gpus all \
    --shm-size=100g \
    -v $(pwd):/workspace \
    --name zgx_cuda \
    zgx-cuda
```

### 3. Connect Cursor

1. Open Cursor
2. Press **F1** or click the remote indicator (bottom left)
3. Select **"Dev Containers: Attach to Running Container..."**
4. Choose **`zgx_cuda`**
5. Open notebooks from `/workspace/notebooks/`

## Container Management

```bash
# Check status
./build-and-run.sh --status

# Stop and remove container
./build-and-run.sh --stop

# Rebuild image only
./build-and-run.sh --build

# Run only (image must exist)
./build-and-run.sh --run
```

## Available Demos

Once connected to the container, you can run:

| Notebook | Description |
|----------|-------------|
| `notebooks/inference/zgx_nano_unified_memory.ipynb` | Unified Memory benchmark demo |
| `notebooks/inference/zgx_nano_ollama_performance.ipynb` | Ollama inference performance |

## Important Notes

### Shared Memory Size

The `--shm-size=100g` flag is **required** for unified memory operations with large models. Without sufficient shared memory, operations may fail.

### GPU Access

Ensure Docker has GPU access configured:
- NVIDIA Container Toolkit must be installed
- Use `--gpus all` to expose GPUs to the container

### Workspace Mounting

The container mounts the project root to `/workspace`. Any changes made inside the container are persisted to your local filesystem.

## Troubleshooting

### Container won't start

```bash
# Check if container exists
docker ps -a | grep zgx_cuda

# Remove and recreate
docker rm -f zgx_cuda
./build-and-run.sh
```

### GPU not detected

```bash
# Verify NVIDIA Container Toolkit
docker run --rm --gpus all ubuntu nvidia-smi
```

### Cursor can't attach

- Ensure the container is running: `docker ps`
- Install the "Dev Containers" extension in Cursor
- Try restarting Cursor

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Container image definition |
| `build-and-run.sh` | Helper script for container management |
| `README.md` | This documentation |

