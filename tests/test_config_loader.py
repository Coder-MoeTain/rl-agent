"""Tests for configuration loader."""

import pytest

from config_loader import get_env_kwargs, get_per_kwargs, get_ppo_kwargs, load_config
from gym_pentest.rewards import RewardConfig


def test_load_config_default():
    config = load_config()
    assert "environment" in config
    assert "ppo" in config
    assert config["environment"]["base_url"] == "http://localhost:3000"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_get_env_kwargs():
    config = load_config()
    kwargs = get_env_kwargs(config)
    assert kwargs["base_url"] == "http://localhost:3000"
    assert isinstance(kwargs["reward_config"], RewardConfig)


def test_get_ppo_kwargs():
    config = load_config()
    ppo = get_ppo_kwargs(config)
    assert ppo["policy"] == "MlpPolicy"
    assert ppo["learning_rate"] == pytest.approx(0.0003)


def test_get_per_kwargs():
    config = load_config()
    per = get_per_kwargs(config)
    assert per["per_capacity"] == 4096
    assert per["per_alpha"] == 0.6
