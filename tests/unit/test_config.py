"""Tests for configuration utilities."""

import pytest
from pathlib import Path

from zgx_onboard.utils.config import load_config, get_settings


def test_load_config():
    """Test loading configuration from YAML file."""
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / "default_config.yaml"
    config = load_config(str(config_path))
    
    assert isinstance(config, dict)
    assert "hardware" in config
    assert "model" in config
    assert "training" in config


def test_get_settings():
    """Test getting settings from environment."""
    settings = get_settings()
    
    assert settings.device in ["cuda", "cpu", "mps"]
    assert settings.batch_size > 0
    assert settings.seed >= 0

