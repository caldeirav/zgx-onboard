"""Setup script for zgx-onboard package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Dependencies are managed with uv and defined in pyproject.toml
# This setup.py is kept for compatibility but dependencies should be added via uv

setup(
    name="zgx-onboard",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Local AI experiments with HP ZGX Nano AI Station",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/zgx-onboard",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    # Dependencies are managed with uv - add packages using: uv add <package>
    install_requires=[],
    extras_require={
        # Development dependencies can be added with: uv add --dev <package>
        "dev": [],
    },
)

