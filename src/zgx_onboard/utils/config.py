"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseModel):
    """Base configuration model."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default config.

    Returns:
        Dictionary containing configuration.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent / "configs" / "default_config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Hardware
    device: str = Field(default="cuda", description="Device to use (cuda, cpu, mps)")
    cuda_visible_devices: str = Field(default="0", description="CUDA device IDs")

    # Paths
    model_cache_dir: str = Field(default="./models/pretrained", description="Model cache directory")
    checkpoint_dir: str = Field(default="./models/checkpoints", description="Checkpoint directory")
    data_dir: str = Field(default="./data", description="Data directory")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: str = Field(default="./logs", description="Log directory")

    # Training
    batch_size: int = Field(default=32, description="Batch size")
    num_workers: int = Field(default=4, description="Number of data loader workers")
    seed: int = Field(default=42, description="Random seed")


def get_settings() -> Settings:
    """Get application settings from environment."""
    return Settings()

