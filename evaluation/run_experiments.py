"""Research-grade evaluation runner with multi-seed support."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from agents.baselines import RandomAgent, RuleBasedAgent
from config_loader import get_env_kwargs, load_config
from evaluation.benchmark_report import generate_benchmark_report
from evaluation.metrics import AggregateMetrics, EpisodeMetrics, compute_aggregate
from evaluation.plots import (
    plot_comparison,
    plot_coverage,
    plot_reward_distribution,
    plot_seed_variance,
    plot_vulnerability_discovery,
)
from evaluation.statistics import compare_algorithms
from gym_pentest.env import PentestEnv
from setup_logging import setup_logging
from utils.results import save_aggregate_csv, save_episodes_csv, save_results_json
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
        forms_found=info.get("forms_found", 0),
        params_found=info.get("params_found", 0),
        duplicate_findings=info.get("duplicate_actions", 0),
        report_generated=info.get("report_generated", False),
        requests_made=info.get("episode_requests", steps),
        challenges_solved=info.get("challenges_solved", 0),
        scoreboard_progress=info.get("scoreboard_progress", 0.0),
        steps_to_first_challenge=info.get("steps_to_first_challenge"),
        vuln_types=info.get("vuln_types", []),
    )


def evaluate_algorithm(
    algorithm: str,
    config: dict[str, Any],
    seeds: list[int],
    episodes_per_seed: int,
    model_path: str | None = None,
    mock: bool = False,
    ablation: str | None = None,
) -> tuple[list[EpisodeMetrics], float | None]:
    """Evaluate one algorithm across multiple seeds."""
    env_kwargs = get_env_kwargs(config)
    if mock:
        from gym_pentest.mock_http import build_mock_http_client
        from gym_pentest.scoreboard import MockScoreboard

        env_kwargs.update(
            {
                "http_client": build_mock_http_client(),
                "scoreboard": MockScoreboard(),
                "disable_scope_guard": True,
                "disable_safety_controls": True,
                "max_steps": min(env_kwargs.get("max_steps", 100), 30),
            }
        )
    all_episodes: list[EpisodeMetrics] = []
    training_time: float | None = None

    for seed in seeds:
        set_global_seed(seed)
        env = PentestEnv(**env_kwargs)

        if algorithm == "random":
            agent = RandomAgent(env.action_space.n, seed=seed)
        elif algorithm == "rule_based":
            agent = RuleBasedAgent(env.action_space.n, seed=seed)
        elif algorithm == "multi_agent":
            from agents.multi_agent_framework import MultiAgentRLAgent

            agent = MultiAgentRLAgent(seed=seed)
        elif algorithm in ("ppo", "ppo_per"):
            from stable_baselines3 import PPO

            if model_path is None:
                model_dir = Path(config.get("training", {}).get("model_dir", "./models/"))
                default_name = "ppo_baseline" if algorithm == "ppo" else "ppo_per_model"
                model_path = str(model_dir / default_name)
            if not Path(model_path).exists():
                if mock:
                    agent = RandomAgent(env.action_space.n, seed=seed)
                else:
                    raise FileNotFoundError(
                        f"Model not found: {model_path}. Train first with "
                        f"python -m agents.train --algo {algorithm}"
                    )
            else:
                agent = PPO.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        for ep in range(episodes_per_seed):
            metrics = run_episode(env, agent, seed, ep, algorithm)
            if ablation:
                metrics.ablation = ablation
            all_episodes.append(metrics)

        env.close()

    return all_episodes, training_time


def run_experiments(config: dict[str, Any] | None = None, mock: bool = False) -> pd.DataFrame:
    """Run full evaluation suite and save results."""
    if config is None:
        config = load_config()

    eval_cfg = config.get("evaluation", {})
    seeds = eval_cfg.get("seeds", [42, 123, 456])
    episodes_per_seed = eval_cfg.get("episodes_per_seed", 10)
    output_dir = Path(eval_cfg.get("output_dir", "./results/"))
    algorithms = eval_cfg.get("algorithms", ["random", "rule_based"])

    output_dir.mkdir(parents=True, exist_ok=True)
    all_episodes: list[EpisodeMetrics] = []
    aggregates: list[AggregateMetrics] = []

    for algo in algorithms:
        print(f"Evaluating {algo}...")
        start = time.time()
        episodes, _ = evaluate_algorithm(algo, config, seeds, episodes_per_seed, mock=mock)
        elapsed = time.time() - start
        all_episodes.extend(episodes)
        agg = compute_aggregate(episodes, algo, training_time=elapsed)
        aggregates.append(agg)
        print(f"  {algo}: mean_reward={agg.mean_reward:.2f}, success_rate={agg.success_rate:.1%}")

    # Save results via utility functions
    save_episodes_csv(all_episodes, output_dir / "episodes.csv")
    save_aggregate_csv(aggregates, output_dir / "aggregate.csv")
    save_results_json(
        all_episodes,
        aggregates,
        {"seeds": seeds, "episodes_per_seed": episodes_per_seed},
        output_dir / "results.json",
    )

    episodes_df = pd.DataFrame([e.__dict__ for e in all_episodes])
    agg_df = pd.DataFrame([a.to_dict() for a in aggregates])

    # Generate plots
    plot_comparison(agg_df, output_dir)
    plot_reward_distribution(episodes_df, output_dir)
    plot_seed_variance(episodes_df, output_dir)
    plot_coverage(episodes_df, output_dir)
    plot_vulnerability_discovery(episodes_df, output_dir)

    # Statistical significance tests
    rewards_by_algo = {
        algo: episodes_df[episodes_df["algorithm"] == algo]["total_reward"].tolist()
        for algo in episodes_df["algorithm"].unique()
    }
    significance = compare_algorithms(rewards_by_algo)
    sig_rows = [s.__dict__ for s in significance]
    pd.DataFrame(sig_rows).to_csv(output_dir / "significance.csv", index=False)

    generate_benchmark_report(episodes_df, agg_df, significance, output_dir / "BENCHMARK.md")

    print(f"\nResults saved to {output_dir}")
    return agg_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run defensive vulnerability assessment RL evaluation experiments"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Algorithms to evaluate",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mocked HTTP/scoreboard (no Docker required)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.algorithms:
        config.setdefault("evaluation", {})["algorithms"] = args.algorithms

    log_cfg = config.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"))

    run_experiments(config, mock=args.mock)


if __name__ == "__main__":
    main()
