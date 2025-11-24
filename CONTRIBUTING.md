# Contributing to ZGX Onboard

Thank you for your interest in contributing to ZGX Onboard! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/zgx-onboard.git`
3. Set up the development environment (see README.md)
4. Create a branch for your changes: `git checkout -b feature/your-feature-name`

## Development Workflow

1. **Make your changes** in your feature branch
2. **Write tests** for new functionality
3. **Run tests** to ensure everything passes: `pytest`
4. **Format code**: `make format`
5. **Check linting**: `make lint`
6. **Commit changes** with clear, descriptive commit messages
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Open a Pull Request** on GitHub

## Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting a PR
- Aim for good test coverage
- Include integration tests for complex workflows

## Commit Messages

Use clear, descriptive commit messages:

```
feat: Add support for new model architecture
fix: Resolve memory leak in inference pipeline
docs: Update README with new examples
test: Add tests for configuration loading
refactor: Simplify agent training loop
```

## Pull Request Process

1. Ensure your code follows the project's style guidelines
2. Update documentation if needed
3. Add tests for new features
4. Ensure all tests pass
5. Update CHANGELOG.md if applicable
6. Request review from maintainers

## Questions?

If you have questions, please open an issue on GitHub or contact the maintainers.

