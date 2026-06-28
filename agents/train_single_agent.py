"""Single agent PPO training script (config-driven)."""

from agents.train import train_ppo
from config_loader import load_config
from setup_logging import setup_logging


def main() -> None:
    config = load_config()
    setup_logging(level=config.get("logging", {}).get("level", "INFO"))
    model_dir = config.get("training", {}).get("model_dir", "./models/")
    train_ppo(config, f"{model_dir}/ppo_baseline")


if __name__ == "__main__":
    main()
