"""Ablation study runner for research experiments."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from config_loader import apply_ablation, load_config
from evaluation.metrics import AggregateMetrics, EpisodeMetrics, compute_aggregate
from evaluation.run_experiments import evaluate_algorithm
from setup_logging import setup_logging
from utils.results import save_aggregate_csv, save_episodes_csv, save_results_json


def run_ablation_study(
    config: dict[str, Any] | None = None,
    mock: bool = False,
    ablation_names: list[str] | None = None,
) -> pd.DataFrame:
    """Run all configured ablation studies and save comparison results."""
    if config is None:
        config = load_config()

    eval_cfg = config.get("evaluation", {})
    seeds = eval_cfg.get("seeds", [42, 123, 456])
    episodes_per_seed = eval_cfg.get("episodes_per_seed", 5)
    output_dir = Path(eval_cfg.get("output_dir", "./results/")) / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    ablations = config.get("ablations", [{"name": "full"}])
    if ablation_names:
        ablations = [a for a in ablations if a.get("name") in ablation_names]

    all_episodes: list[EpisodeMetrics] = []
    aggregates: list[AggregateMetrics] = []

    for ablation in ablations:
        name = ablation.get("name", "unknown")
        print(f"Running ablation: {name}")
        ablated_config = apply_ablation(config, name)
        algo = "ppo_per" if ablation.get("use_per", True) else "ppo"
        if ablation.get("use_multi_agent", False):
            algo = "multi_agent"

        start = time.time()
        episodes, _ = evaluate_algorithm(
            algo, ablated_config, seeds, episodes_per_seed, mock=mock, ablation=name
        )
        elapsed = time.time() - start

        for ep in episodes:
            ep.ablation = name
        all_episodes.extend(episodes)
        agg = compute_aggregate(episodes, algo, training_time=elapsed, ablation=name)
        aggregates.append(agg)
        print(f"  {name}: mean_reward={agg.mean_reward:.2f}, coverage={agg.mean_endpoint_coverage:.1%}")

    save_episodes_csv(all_episodes, output_dir / "ablation_episodes.csv")
    save_aggregate_csv(aggregates, output_dir / "ablation_aggregate.csv")
    save_results_json(
        all_episodes,
        aggregates,
        {"ablations": [a.get("name") for a in ablations], "seeds": seeds},
        output_dir / "ablation_results.json",
    )

    agg_df = pd.DataFrame([a.to_dict() for a in aggregates])
    agg_df.to_csv(output_dir / "ablation_comparison.csv", index=False)
    print(f"\nAblation results saved to {output_dir}")
    return agg_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--ablations", nargs="+", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(level=config.get("logging", {}).get("level", "INFO"))
    run_ablation_study(config, mock=args.mock, ablation_names=args.ablations)


if __name__ == "__main__":
    main()
