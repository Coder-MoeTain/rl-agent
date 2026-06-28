"""Benchmark smoke test."""

from evaluation.run_experiments import run_experiments


def test_benchmark_smoke_mock():
    """Run minimal offline benchmark without Docker."""
    agg_df = run_experiments(
        config={
            "environment": {"max_steps": 10},
            "evaluation": {
                "seeds": [42],
                "episodes_per_seed": 1,
                "algorithms": ["random", "rule_based", "multi_agent"],
                "output_dir": "./results/test_smoke/",
            },
            "safety": {"safe_mode": False},
        },
        mock=True,
    )
    assert len(agg_df) == 3
    assert "mean_reward" in agg_df.columns
