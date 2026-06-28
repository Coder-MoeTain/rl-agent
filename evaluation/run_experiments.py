"""Research-grade evaluation runner with multi-seed support."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from agents.baselines import RandomAgent, RuleBasedAgent
from config_loader import get_env_kwargs, load_config
from evaluation.metrics import AggregateMetrics, EpisodeMetrics, compute_aggregate
from evaluation.plots import plot_comparison, plot_reward_distribution, plot_seed_variance
from gym_pentest.env import PentestEnv
from setup_logging import setup_logging
from utils.seeds import set_global_seed


def run_episode(
    env: PentestEnv,
    agent: Any,
    seed: int,
    episode_idx: int,
    algorithm: str,
    deterministic: bool = True,
) -> EpisodeMetrics:
    """Run a single evaluation episode."""
    obs, _ = env.reset(seed=seed + episode_idx)
    if hasattr(agent, "reset"):
        agent.reset()

    done = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (done or truncated):
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(int(action))
        total_reward += reward
        steps += 1

    return EpisodeMetrics(
        seed=seed,
        episode=episode_idx,
        algorithm=algorithm,
        total_reward=total_reward,
        episode_length=steps,
        vulnerabilities=info.get("vulnerabilities", 0),
        confirmed_vulnerabilities=info.get("confirmed_vulnerabilities", 0),
        endpoints_discovered=info.get("discovered_count", 0),
        endpoint_coverage=info.get("endpoint_coverage", 0.0),
        steps_to_first_finding=info.get("steps_to_first_finding"),
        success=total_reward > 0,
        logged_in=info.get("logged_in", False),
        vuln_types=info.get("vuln_types", []),
    )


def evaluate_algorithm(
    algorithm: str,
    config: Dict[str, Any],
    seeds: List[int],
    episodes_per_seed: int,
    model_path: Optional[str] = None,
) -> tuple[List[EpisodeMetrics], Optional[float]]:
    """Evaluate one algorithm across multiple seeds."""
    env_kwargs = get_env_kwargs(config)
    all_episodes: List[EpisodeMetrics] = []
    training_time: Optional[float] = None

    for seed in seeds:
        set_global_seed(seed)
        env = PentestEnv(**env_kwargs)

        if algorithm == "random":
            agent = RandomAgent(env.action_space.n, seed=seed)
        elif algorithm == "rule_based":
            agent = RuleBasedAgent(env.action_space.n, seed=seed)
        elif algorithm in ("ppo", "ppo_per"):
            from stable_baselines3 import PPO

            if model_path is None:
                model_dir = Path(config.get("training", {}).get("model_dir", "./models/"))
                default_name = "ppo_baseline" if algorithm == "ppo" else "ppo_per_model"
                model_path = str(model_dir / default_name)
            agent = PPO.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        for ep in range(episodes_per_seed):
            metrics = run_episode(env, agent, seed, ep, algorithm)
            all_episodes.append(metrics)

        env.close()

    return all_episodes, training_time


def run_experiments(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Run full evaluation suite and save results."""
    if config is None:
        config = load_config()

    eval_cfg = config.get("evaluation", {})
    seeds = eval_cfg.get("seeds", [42, 123, 456])
    episodes_per_seed = eval_cfg.get("episodes_per_seed", 10)
    output_dir = Path(eval_cfg.get("output_dir", "./results/"))
    algorithms = eval_cfg.get("algorithms", ["random", "rule_based"])

    output_dir.mkdir(parents=True, exist_ok=True)
    all_episodes: List[EpisodeMetrics] = []
    aggregates: List[AggregateMetrics] = []

    for algo in algorithms:
        print(f"Evaluating {algo}...")
        start = time.time()
        episodes, _ = evaluate_algorithm(algo, config, seeds, episodes_per_seed)
        elapsed = time.time() - start
        all_episodes.extend(episodes)
        agg = compute_aggregate(episodes, algo, training_time=elapsed)
        aggregates.append(agg)
        print(f"  {algo}: mean_reward={agg.mean_reward:.2f}, success_rate={agg.success_rate:.1%}")

    # Save episode-level CSV
    episodes_df = pd.DataFrame([e.__dict__ for e in all_episodes])
    episodes_df.to_csv(output_dir / "episodes.csv", index=False)

    # Save aggregate CSV
    agg_df = pd.DataFrame([a.to_dict() for a in aggregates])
    agg_df.to_csv(output_dir / "aggregate.csv", index=False)

    # Save JSON
    results_json = {
        "config": {"seeds": seeds, "episodes_per_seed": episodes_per_seed},
        "aggregate": [a.to_dict() for a in aggregates],
        "episodes": [e.__dict__ for e in all_episodes],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    # Generate plots
    plot_comparison(agg_df, output_dir)
    plot_reward_distribution(episodes_df, output_dir)
    plot_seed_variance(episodes_df, output_dir)

    print(f"\nResults saved to {output_dir}")
    return agg_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pentesting RL evaluation experiments")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Algorithms to evaluate",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.algorithms:
        config.setdefault("evaluation", {})["algorithms"] = args.algorithms

    log_cfg = config.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"))

    run_experiments(config)


if __name__ == "__main__":
    main()
