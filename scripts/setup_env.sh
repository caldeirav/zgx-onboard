#!/bin/bash
# Setup script for ZGX Onboard development environment using uv

set -e

echo "Setting up ZGX Onboard development environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Using uv version: $(uv --version)"

# Initialize project with uv (creates .venv and uv.lock if needed)
echo "Initializing project with uv..."
uv sync

# Install package in development mode
echo "Installing package in development mode..."
uv pip install -e .

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Please update .env with your configuration"
    else
        echo "Warning: .env.example not found"
    fi
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p mlruns
mkdir -p tensorboard_logs

echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Or use uv to run commands directly:"
echo "  uv run python <script.py>"
echo ""
echo "To add dependencies, use:"
echo "  uv add <package>"
echo "  uv add --dev <package>  # for development dependencies"

