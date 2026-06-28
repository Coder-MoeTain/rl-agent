"""Training script for PPO with Prioritized Experience Replay (config-driven)."""

from agents.train import train_ppo_per
from config_loader import load_config
from setup_logging import setup_logging

if __name__ == "__main__":
    config = load_config()
    setup_logging(level=config.get("logging", {}).get("level", "INFO"))
    model_dir = config.get("training", {}).get("model_dir", "./models/")
    train_ppo_per(config, f"{model_dir}/ppo_per_model")
