.PHONY: help install install-dev test lint format clean setup

help:
	@echo "Available commands:"
	@echo "  make setup       - Set up development environment"
	@echo "  make install     - Install package and dependencies"
	@echo "  make install-dev - Install with development dependencies"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting checks"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean build artifacts"

setup:
	@./scripts/setup_env.sh

install:
	uv sync
	uv pip install -e .

install-dev:
	uv sync
	uv pip install -e .

test:
	uv run pytest tests/ --cov=zgx_onboard --cov-report=term-missing

lint:
	uv run flake8 src/ tests/
	uv run mypy src/ --ignore-missing-imports
	uv run black --check src/ tests/
	uv run isort --check-only src/ tests/

format:
	uv run black src/ tests/
	uv run isort src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

