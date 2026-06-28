"""Generate benchmark report from experiment results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.metrics import AggregateMetrics
from evaluation.statistics import compare_algorithms, summarize_with_ci


def generate_benchmark_report(
    episodes_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    significance_results: list[Any],
    output_path: Path,
) -> Path:
    """Write BENCHMARK.md with published results table and significance tests."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Results",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Aggregate Performance",
        "",
        "| Algorithm | Mean Reward | Std | 95% CI | Success Rate | Endpoint Coverage | Vuln Rate | Steps to 1st Finding |",
        "|-----------|-------------|-----|--------|--------------|-------------------|-----------|----------------------|",
    ]

    for _, row in aggregate_df.iterrows():
        algo = row["algorithm"]
        rewards = episodes_df[episodes_df["algorithm"] == algo]["total_reward"].tolist()
        ci = summarize_with_ci(rewards, algo)
        ci_str = f"[{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]"
        steps = row.get("mean_steps_to_first_finding", float("inf"))
        steps_str = f"{steps:.1f}" if steps != float("inf") else "N/A"
        lines.append(
            f"| {algo} | {row['mean_reward']:.2f} | {row['std_reward']:.2f} | {ci_str} | "
            f"{row['success_rate']:.1%} | {row['mean_endpoint_coverage']:.1%} | "
            f"{row['mean_vuln_discovery_rate']:.2f} | {steps_str} |"
        )

    lines.extend(["", "## Statistical Significance (Welch t-test, α=0.05)", ""])
    if significance_results:
        lines.append(
            "| Algorithm A | Algorithm B | Mean A | Mean B | p-value | Significant | Cohen's d |"
        )
        lines.append(
            "|-------------|-------------|--------|--------|---------|-------------|-----------|"
        )
        for r in significance_results:
            d = f"{r.effect_size:.3f}" if r.effect_size is not None else "N/A"
            sig = "✓" if r.significant else "✗"
            lines.append(
                f"| {r.algorithm_a} | {r.algorithm_b} | {r.mean_a:.2f} | {r.mean_b:.2f} | "
                f"{r.p_value:.4f} | {sig} | {d} |"
            )
    else:
        lines.append("_No pairwise comparisons available._")

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "docker compose up -d",
            "python -m evaluation.run_experiments --algorithms random rule_based",
            "```",
            "",
        ]
    )

    output_path.write_text("\n".join(lines))
    return output_path
