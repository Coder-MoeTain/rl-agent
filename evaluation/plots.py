"""Plot generation for experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def plot_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate comparison plots across algorithms."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("mean_reward", "Mean Episode Reward"),
        ("mean_endpoint_coverage", "Endpoint Coverage"),
        ("mean_vuln_discovery_rate", "Vulnerability Discovery Rate"),
        ("success_rate", "Success Rate"),
        ("mean_steps_to_first_finding", "Steps to First Finding"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        if col in results_df.columns:
            results_df.plot(x="algorithm", y=col, kind="bar", ax=ax, legend=False, color="steelblue")
            ax.set_title(title)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)

    # Training time if available
    ax = axes[5]
    if "training_time_seconds" in results_df.columns and results_df["training_time_seconds"].notna().any():
        results_df.plot(x="algorithm", y="training_time_seconds", kind="bar", ax=ax, legend=False, color="coral")
        ax.set_title("Training Time (seconds)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / "algorithm_comparison.png", dpi=150)
    plt.close(fig)


def plot_reward_distribution(episodes_df: pd.DataFrame, output_dir: Path) -> None:
    """Box plot of reward distribution per algorithm."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if episodes_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    algorithms = episodes_df["algorithm"].unique()
    data = [episodes_df[episodes_df["algorithm"] == a]["total_reward"].values for a in algorithms]
    ax.boxplot(data, labels=algorithms)
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Distribution by Algorithm")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(output_dir / "reward_distribution.png", dpi=150)
    plt.close(fig)


def plot_seed_variance(episodes_df: pd.DataFrame, output_dir: Path) -> None:
    """Line plot showing reward variance across seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if episodes_df.empty or "seed" not in episodes_df.columns:
        return

    grouped = episodes_df.groupby(["algorithm", "seed"])["total_reward"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in grouped["algorithm"].unique():
        subset = grouped[grouped["algorithm"] == algo]
        ax.plot(subset["seed"], subset["total_reward"], marker="o", label=algo)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward Stability Across Seeds")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "seed_variance.png", dpi=150)
    plt.close(fig)
