"""Configuration loader for production autonomous pentester."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from gym_pentest.rewards import RewardConfig
from gym_pentest.safety import SafetyConfig

ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "config.yaml"
PRODUCTION_CONFIG = ROOT / "config.production.yaml"


def resolve_config_path(config_path: str | None = None, production_default: bool = False) -> Path:
    """Resolve config path: explicit > PENTESTER_CONFIG env > production (if flagged) > default."""
    if config_path:
        return Path(config_path)
    env_path = os.environ.get("PENTESTER_CONFIG")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    if production_default and PRODUCTION_CONFIG.exists():
        return PRODUCTION_CONFIG
    return DEFAULT_CONFIG


def load_config(config_path: str | None = None, production_default: bool = False) -> dict[str, Any]:
    """Load configuration from YAML file."""
    path = resolve_config_path(config_path, production_default=production_default)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config: dict[str, Any] = yaml.safe_load(f)
    config["_config_path"] = str(path)
    return config


def get_production_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return production-specific settings with defaults."""
    prod = config.get("production", {})
    return {
        "default_algorithm": prod.get("default_algorithm", "auto"),
        "output_dir": prod.get("output_dir", "./reports/"),
        "health_check_timeout": float(prod.get("health_check_timeout", 5.0)),
        "engagement_timeout_seconds": float(prod.get("engagement_timeout_seconds", 600.0)),
        "retry_max": int(prod.get("retry_max", 3)),
        "circuit_breaker_threshold": int(prod.get("circuit_breaker_threshold", 5)),
    }


def get_mock_env_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build PentestEnv kwargs with mocked HTTP for offline production smoke tests."""
    from gym_pentest.mock_http import build_mock_http_client
    from gym_pentest.scoreboard import MockScoreboard

    env_kwargs = get_env_kwargs(config)
    env_kwargs.update(
        {
            "http_client": build_mock_http_client(),
            "scoreboard": MockScoreboard(),
            "disable_scope_guard": True,
            "disable_safety_controls": True,
            "max_steps": min(env_kwargs.get("max_steps", 100), 30),
        }
    )
    return env_kwargs


def get_safety_config(config: dict[str, Any]) -> SafetyConfig:
    """Build SafetyConfig from YAML."""
    safety_cfg = config.get("safety", {})
    allowed_hosts = safety_cfg.get("allowed_hosts")
    return SafetyConfig(
        safe_mode=safety_cfg.get("safe_mode", True),
        allow_public_internet=safety_cfg.get("allow_public_internet", False),
        allow_private_networks=safety_cfg.get("allow_private_networks", True),
        max_requests_per_episode=safety_cfg.get("max_requests_per_episode", 200),
        requests_per_second=safety_cfg.get("requests_per_second", 5.0),
        emergency_stop_on_scope_violation=safety_cfg.get(
            "emergency_stop_on_scope_violation", True
        ),
        max_scope_violations=safety_cfg.get("max_scope_violations", 3),
        log_all_actions=safety_cfg.get("log_all_actions", True),
        allowed_hosts=frozenset(allowed_hosts) if allowed_hosts else SafetyConfig().allowed_hosts,
    )


def get_reward_config(config: dict[str, Any]) -> RewardConfig:
    """Build RewardConfig from YAML."""
    reward_cfg = config.get("rewards", {})
    return RewardConfig(
        **{k: v for k, v in reward_cfg.items() if hasattr(RewardConfig, k)}
    )


def get_env_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract PentestEnv constructor kwargs from config."""
    env_cfg = config.get("environment", {})
    return {
        "base_url": env_cfg.get("base_url", "http://localhost:3000"),
        "max_steps": env_cfg.get("max_steps", 100),
        "mission_vuln_threshold": env_cfg.get("mission_vuln_threshold", 2),
        "mission_challenge_threshold": env_cfg.get("mission_challenge_threshold", 1),
        "use_scoreboard": env_cfg.get("use_scoreboard", True),
        "reward_config": get_reward_config(config),
        "safety_config": get_safety_config(config),
    }


def get_ppo_kwargs(config: dict[str, Any]) -> dict[str, Any]:
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


def get_per_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract PER hyperparameters from config."""
    per = config.get("per", {})
    return {
        "per_capacity": per.get("capacity", 4096),
        "per_alpha": per.get("alpha", 0.6),
        "per_beta_start": per.get("beta_start", 0.4),
        "per_beta_frames": per.get("beta_frames", 100000),
        "per_batch_size": per.get("batch_size", 64),
    }


def apply_ablation(config: dict[str, Any], ablation_name: str) -> dict[str, Any]:
    """Apply ablation overrides to a copy of config."""
    import copy

    cfg = copy.deepcopy(config)
    ablations = cfg.get("ablations", [])
    for ab in ablations:
        if ab.get("name") == ablation_name:
            rewards = cfg.setdefault("rewards", {})
            for key in (
                "use_attack_graph_features",
                "use_duplicate_penalty",
                "use_vulnerability_confirmation",
                "use_safety_aware_reward",
            ):
                if key in ab:
                    rewards[key] = ab[key]
            if "use_per" in ab:
                cfg.setdefault("evaluation", {})["use_per"] = ab["use_per"]
            if "use_multi_agent" in ab:
                cfg.setdefault("evaluation", {})["use_multi_agent"] = ab["use_multi_agent"]
            break
    return cfg
