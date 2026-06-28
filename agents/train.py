"""Unified training entry point for all RL algorithms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from config_loader import get_env_kwargs, get_per_kwargs, get_ppo_kwargs, load_config
from custom_sb3_per import PPO_PER
from gym_pentest.env import PentestEnv
from setup_logging import setup_logging
from utils.seeds import set_global_seed


def make_env_fn(config: dict):
    """Factory for vectorized environments."""
    env_kwargs = get_env_kwargs(config)

    def _init():
        return PentestEnv(**env_kwargs)

    return _init


def train_ppo(config: dict, model_path: str) -> None:
    """Train PPO baseline."""
    training = config.get("training", {})
    ppo_kwargs = get_ppo_kwargs(config)
    seed = training.get("seed", 42)
    set_global_seed(seed)

    n_envs = config.get("ppo", {}).get("n_envs", 4)
    env = make_vec_env(make_env_fn(config), n_envs=n_envs)

    model = PPO(
        env=env,
        verbose=1,
        tensorboard_log=training.get("tensorboard_log", "./tensorboard_logs/"),
        seed=seed,
        **ppo_kwargs,
    )
    model.learn(total_timesteps=training.get("total_timesteps", 20000))
    model.save(model_path)
    print(f"Saved {model_path}")


def train_ppo_per(config: dict, model_path: str) -> None:
    """Train PPO with prioritized experience replay."""
    training = config.get("training", {})
    ppo_kwargs = get_ppo_kwargs(config)
    per_kwargs = get_per_kwargs(config)
    seed = training.get("seed", 42)
    set_global_seed(seed)

    n_envs = config.get("ppo", {}).get("n_envs", 4)
    env = make_vec_env(make_env_fn(config), n_envs=n_envs)

    model = PPO_PER(
        env=env,
        verbose=1,
        tensorboard_log=training.get("tensorboard_log", "./tensorboard_logs/"),
        seed=seed,
        **ppo_kwargs,
        **per_kwargs,
    )
    model.learn(total_timesteps=training.get("total_timesteps", 20000))
    model.save(model_path)
    print(f"Saved {model_path}")


def train_multi_agent(config: dict) -> None:
    """Train multi-agent recon/exploit system."""
    from agents.train_multi_agent_per_is import train as multi_train

    multi_train(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pentesting RL agents")
    parser.add_argument(
        "--algo",
        choices=["ppo", "ppo_per", "multi"],
        default="ppo",
        help="Algorithm to train",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Output model path")
    args = parser.parse_args()

    config = load_config(args.config)
    log_cfg = config.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("file"))

    model_dir = Path(config.get("training", {}).get("model_dir", "./models/"))
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.algo == "ppo":
        path = args.model or str(model_dir / "ppo_baseline")
        train_ppo(config, path)
    elif args.algo == "ppo_per":
        path = args.model or str(model_dir / "ppo_per_model")
        train_ppo_per(config, path)
    elif args.algo == "multi":
        train_multi_agent(config)


if __name__ == "__main__":
    main()
