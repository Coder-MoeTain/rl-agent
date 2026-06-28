"""Tests for benchmark report generation."""

from pathlib import Path

import pandas as pd

from evaluation.benchmark_report import generate_benchmark_report
from evaluation.statistics import compare_algorithms


def test_generate_benchmark_report(tmp_path):
    episodes_df = pd.DataFrame(
        {
            "algorithm": ["random", "random", "rule_based", "rule_based"],
            "total_reward": [1.0, 2.0, 10.0, 12.0],
            "success_rate": [True, True, True, True],
        }
    )
    aggregate_df = pd.DataFrame(
        {
            "algorithm": ["random", "rule_based"],
            "mean_reward": [1.5, 11.0],
            "std_reward": [0.5, 1.0],
            "success_rate": [1.0, 1.0],
            "mean_endpoint_coverage": [0.1, 0.5],
            "mean_vuln_discovery_rate": [0.0, 1.0],
            "mean_steps_to_first_finding": [float("inf"), 5.0],
        }
    )
    rewards_by_algo = {
        "random": [1.0, 2.0],
        "rule_based": [10.0, 12.0],
    }
    significance = compare_algorithms(rewards_by_algo)
    path = generate_benchmark_report(
        episodes_df, aggregate_df, significance, tmp_path / "BENCHMARK.md"
    )
    content = path.read_text()
    assert "Benchmark Results" in content
    assert "rule_based" in content
    assert "Statistical Significance" in content
