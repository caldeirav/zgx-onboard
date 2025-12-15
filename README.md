# ZGX Onboard

Code repository for local AI experiments with the HP ZGX Nano AI Station. This project showcases what can be done with the hardware locally in terms of developing AI systems, including model inference, AI agents, model tuning, and reinforcement learning.

## Features

- **Model Inference**: Run various AI models locally on the ZGX Nano AI Station
- **AI Agents**: Develop and experiment with intelligent agents
- **Model Tuning**: Hyperparameter optimization and fine-tuning capabilities
- **Reinforcement Learning**: RL experiments and training pipelines
- **Experiment Tracking**: Integration with MLflow and TensorBoard

## Requirements

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
- CUDA-capable GPU (for GPU acceleration)
- Linux (tested on Ubuntu)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/zgx-onboard.git
cd zgx-onboard
```

### 2. Install uv

If you don't have `uv` installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Set Up Environment

#### Option A: Using the Setup Script

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

#### Option B: Manual Setup with uv

```bash
# Initialize project and create virtual environment
uv sync

# Install package in development mode
uv pip install -e .

# Activate virtual environment (optional, uv can run commands directly)
source .venv/bin/activate
```

### 4. Add Dependencies

This project uses `uv` for dependency management. Add packages as needed:

```bash
# Add a package
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Examples:
uv add numpy torch transformers
uv add --dev pytest black
```

The dependencies will be automatically added to `pyproject.toml` and `uv.lock` will be updated.

### 5. Configure Environment

Copy the example environment file and update with your settings:

```bash
cp env.example .env
# Edit .env with your configuration
```

### 6. Verify Installation

```bash
# Using uv (recommended)
uv run python -c "import zgx_onboard; print('Installation successful!')"

# Or with activated virtual environment
source .venv/bin/activate
python -c "import zgx_onboard; print('Installation successful!')"
```

## CUDA Dev Container

For demos requiring CUDA (e.g., Unified Memory on Grace-Blackwell architecture), use the provided Docker container:

### Quick Start with Docker

```bash
# 1. Create environment file with HuggingFace token
echo "HF_TOKEN=hf_your_token_here" > .env

# 2. Build and run the container
cd docker
./build-and-run.sh

# 3. Attach Cursor to the container
#    - Press F1 → "Dev Containers: Attach to Running Container..."
#    - Select "zgx_cuda"
```

### Container Features

- **CUDA 12.6** development environment (Ubuntu 24.04)
- **Nsight Systems 2024.5** for profiling
- Pre-installed ML stack (PyTorch, Transformers, Accelerate, BitsAndBytes)
- **nvitop** for real-time GPU monitoring

See [docker/README.md](docker/README.md) for detailed container documentation.

## Available Demos

| Demo | Location | Description |
|------|----------|-------------|
| **Unified Memory Benchmark** | `notebooks/inference/zgx_nano_unified_memory.ipynb` | CUDA 13 unified memory demo with TTFT/TPS metrics |
| **Ollama Performance** | `notebooks/inference/zgx_nano_ollama_performance.ipynb` | Local LLM inference benchmarking |

## Project Structure

```
zgx-onboard/
├── docker/                   # CUDA Dev Container
│   ├── Dockerfile            # Container image definition
│   ├── build-and-run.sh      # Container management script
│   └── README.md             # Container documentation
├── src/
│   └── zgx_onboard/          # Main package
│       ├── inference/         # Model inference modules
│       ├── agents/            # AI agent implementations
│       ├── tuning/            # Model tuning and optimization
│       └── utils/             # Utility functions
│           ├── config.py      # Configuration management
│           └── logging.py     # Logging utilities
├── experiments/               # Experimental code and scripts
│   ├── inference/             # Inference experiments
│   ├── agents/                # Agent experiments
│   ├── tuning/                # Tuning experiments
│   └── reinforcement_learning/ # RL experiments
├── notebooks/                 # Jupyter notebooks
│   ├── inference/             # Inference examples
│   ├── agents/                # Agent demonstrations
│   ├── tuning/                # Tuning examples
│   └── reinforcement_learning/ # RL experiments
├── data/                      # Data storage
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed data files
├── models/                    # Model storage
│   ├── checkpoints/           # Training checkpoints
│   └── pretrained/            # Pretrained models
├── configs/                   # Configuration files
│   └── default_config.yaml    # Default configuration
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
│   └── setup_env.sh           # Environment setup script
├── pyproject.toml             # Project metadata and dependencies (managed with uv)
├── uv.lock                    # Lock file for reproducible builds (generated by uv)
├── setup.py                   # Package setup script
└── README.md                  # This file
```

## Usage

### Running Inference

```python
from zgx_onboard.inference import run_inference

# Run inference with a model
results = run_inference(model_name="your_model", input_data="your_data")
```

### Training an Agent

```python
from zgx_onboard.agents import train_agent

# Train an agent
agent = train_agent(
    env_name="CartPole-v1",
    algorithm="PPO",
    total_timesteps=100000
)
```

### Model Tuning

```python
from zgx_onboard.tuning import optimize_hyperparameters

# Optimize hyperparameters
best_params = optimize_hyperparameters(
    model_class=YourModel,
    n_trials=100,
    backend="optuna"
)
```

### Using Configuration

```python
from zgx_onboard import load_config

# Load configuration
config = load_config("configs/default_config.yaml")

# Access configuration values
device = config["hardware"]["device"]
batch_size = config["training"]["batch_size"]
```

## Dependency Management

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

### Adding Dependencies

```bash
# Add a runtime dependency
uv add <package-name>

# Add with version constraint
uv add "numpy>=1.24.0"

# Add multiple packages
uv add torch transformers accelerate

# Add a development dependency
uv add --dev pytest black flake8
```

### Removing Dependencies

```bash
uv remove <package-name>
```

### Syncing Dependencies

After cloning the repository or when dependencies change:

```bash
uv sync
```

This will:
- Create a virtual environment (`.venv`) if it doesn't exist
- Install all dependencies from `pyproject.toml`
- Update `uv.lock` for reproducible builds

### Running Commands with uv

You can run Python scripts directly with uv without activating the virtual environment:

```bash
uv run python script.py
uv run pytest
uv run jupyter lab
```

## Configuration

The project uses YAML configuration files for easy customization. See `configs/default_config.yaml` for all available options.

Key configuration sections:
- **Hardware**: Device settings, CUDA configuration
- **Model**: Model paths and checkpoint settings
- **Training**: Training hyperparameters
- **Inference**: Inference settings
- **Agents**: Agent configuration
- **Reinforcement Learning**: RL algorithm settings
- **Tuning**: Hyperparameter optimization settings
- **Tracking**: Experiment tracking (MLflow, TensorBoard)

## Development

### Running Tests

```bash
# Run all tests (using uv)
uv run pytest

# Or with activated environment
source .venv/bin/activate
pytest

# Run with coverage
uv run pytest --cov=zgx_onboard --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_config.py
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Sort imports
uv run isort src/ tests/

# Lint code
uv run flake8 src/ tests/

# Type checking
uv run mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Experiment Tracking

The project supports multiple experiment tracking tools:

### MLflow

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=./tensorboard_logs
```

## Hardware Considerations

The HP ZGX Nano AI Station is optimized for local AI workloads. When configuring your experiments:

- Use appropriate batch sizes for your GPU memory
- Monitor GPU utilization and temperature
- Consider mixed precision training for efficiency
- Use data loaders with appropriate `num_workers` settings

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HP ZGX Nano AI Station hardware
- Open source AI/ML community

## Support

For issues, questions, or contributions, please open an issue on GitHub.
