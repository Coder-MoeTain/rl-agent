"""Benchmark multi-agent recon/exploit training."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.train_multi_agent_per_is import train as train_multi
from config_loader import load_config


def main() -> None:
    """Run short multi-agent training benchmark."""
    config = load_config()
    ma = config.setdefault("multi_agent", {})
    ma["num_episodes"] = 5
    print("Training multi-agent (short benchmark, 5 episodes)...")
    train_multi(config)
    print("Multi-agent benchmark complete.")


if __name__ == "__main__":
    main()
