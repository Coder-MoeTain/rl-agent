"""Configuration loader for the project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from gym_pentest.rewards import RewardConfig


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to project root config.yaml.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def get_env_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract PentestEnv constructor kwargs from config."""
    env_cfg = config.get("environment", {})
    reward_cfg = config.get("rewards", {})
    reward_config = RewardConfig(**{k: v for k, v in reward_cfg.items() if hasattr(RewardConfig, k)})
    return {
        "base_url": env_cfg.get("base_url", "http://localhost:3000"),
        "max_steps": env_cfg.get("max_steps", 100),
        "mission_vuln_threshold": env_cfg.get("mission_vuln_threshold", 2),
        "reward_config": reward_config,
    }


def get_ppo_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract PPO hyperparameters from config."""
    ppo = config.get("ppo", {})
    return {
        "policy": ppo.get("policy", "MlpPolicy"),
        "learning_rate": ppo.get("learning_rate", 3e-4),
        "n_steps": ppo.get("n_steps", 2048),
        "batch_size": ppo.get("batch_size", 64),
        "n_epochs": ppo.get("n_epochs", 10),
        "gamma": ppo.get("gamma", 0.99),
        "gae_lambda": ppo.get("gae_lambda", 0.95),
        "clip_range": ppo.get("clip_range", 0.2),
        "ent_coef": ppo.get("ent_coef", 0.01),
        "vf_coef": ppo.get("vf_coef", 0.5),
        "max_grad_norm": ppo.get("max_grad_norm", 0.5),
    }


def get_per_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract PER hyperparameters from config."""
    per = config.get("per", {})
    return {
        "per_capacity": per.get("capacity", 4096),
        "per_alpha": per.get("alpha", 0.6),
        "per_beta_start": per.get("beta_start", 0.4),
        "per_beta_frames": per.get("beta_frames", 100000),
        "per_batch_size": per.get("batch_size", 64),
    }
