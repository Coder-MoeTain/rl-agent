"""Plot generation for experiment results."""

from __future__ import annotations

from pathlib import Path

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
            results_df.plot(
                x="algorithm", y=col, kind="bar", ax=ax, legend=False, color="steelblue"
            )
            ax.set_title(title)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)

    # Training time if available
    ax = axes[5]
    if (
        "training_time_seconds" in results_df.columns
        and results_df["training_time_seconds"].notna().any()
    ):
        results_df.plot(
            x="algorithm", y="training_time_seconds", kind="bar", ax=ax, legend=False, color="coral"
        )
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


def plot_coverage(episodes_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of endpoint coverage by algorithm."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if episodes_df.empty:
        return
    grouped = episodes_df.groupby("algorithm")["endpoint_coverage"].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped.plot(kind="bar", ax=ax, color="seagreen")
    ax.set_ylabel("Mean Endpoint Coverage")
    ax.set_title("Endpoint Coverage by Algorithm")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(output_dir / "endpoint_coverage.png", dpi=150)
    plt.close(fig)


def plot_vulnerability_discovery(episodes_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of vulnerability discovery rate by algorithm."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if episodes_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped = episodes_df.groupby("algorithm").agg(
        vulns=("vulnerabilities", "mean"),
        confirmed=("confirmed_vulnerabilities", "mean"),
    )
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean Count per Episode")
    ax.set_title("Vulnerability Discovery by Algorithm")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(["Total Findings", "Confirmed Findings"])
    plt.tight_layout()
    fig.savefig(output_dir / "vulnerability_discovery.png", dpi=150)
    plt.close(fig)


def plot_training_curves(rewards: list[float], output_dir: Path, name: str = "training") -> None:
    """Save training reward curve plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rewards:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rewards, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title(f"Training Curve: {name}")
    plt.tight_layout()
    fig.savefig(output_dir / f"{name}_curve.png", dpi=150)
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
