"""Train all agent configurations using unified training module."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.train import train_multi_agent, train_ppo, train_ppo_per
from config_loader import load_config
from setup_logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Train PPO, PPO+PER, and multi-agent configurations sequentially."""
    config = load_config()
    setup_logging(level=config.get("logging", {}).get("level", "INFO"), log_file="training_all.log")

    model_dir = Path(config.get("training", {}).get("model_dir", "./models/"))
    model_dir.mkdir(parents=True, exist_ok=True)

    training = config.setdefault("training", {})
    original_timesteps = training.get("total_timesteps", 20000)
    training["total_timesteps"] = min(original_timesteps, 10000)

    results: dict[str, bool] = {}

    for name, fn, path in [
        (
            "ppo_baseline",
            lambda: train_ppo(config, str(model_dir / "ppo_baseline")),
            "ppo_baseline",
        ),
        (
            "ppo_per",
            lambda: train_ppo_per(config, str(model_dir / "ppo_per_model")),
            "ppo_per_model",
        ),
        ("multi_agent", lambda: train_multi_agent(config), "multi_agent"),
    ]:
        logger.info("=" * 60)
        logger.info("Training %s", name)
        logger.info("=" * 60)
        try:
            fn()
            results[name] = True
            logger.info("✓ %s complete", name)
        except Exception as exc:
            results[name] = False
            logger.error("✗ %s failed: %s", name, exc)
            logger.error(traceback.format_exc())

    training["total_timesteps"] = original_timesteps
    logger.info("Training summary: %s", results)


if __name__ == "__main__":
    main()
