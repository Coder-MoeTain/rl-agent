"""Tests for result saving utilities."""

from evaluation.metrics import EpisodeMetrics, compute_aggregate
from utils.results import save_aggregate_csv, save_episodes_csv, save_results_json


def test_save_episodes_csv(tmp_path):
    episodes = [
        EpisodeMetrics(
            seed=42,
            episode=0,
            algorithm="random",
            total_reward=10.0,
            episode_length=50,
            vulnerabilities=1,
            confirmed_vulnerabilities=0,
            endpoints_discovered=5,
            endpoint_coverage=0.5,
            steps_to_first_finding=10,
            success=True,
            logged_in=False,
        )
    ]
    path = save_episodes_csv(episodes, tmp_path / "episodes.csv")
    assert path.exists()
    assert "random" in path.read_text()


def test_save_aggregate_csv(tmp_path):
    episodes = [
        EpisodeMetrics(
            seed=42,
            episode=0,
            algorithm="random",
            total_reward=10.0,
            episode_length=50,
            vulnerabilities=1,
            confirmed_vulnerabilities=0,
            endpoints_discovered=5,
            endpoint_coverage=0.5,
            steps_to_first_finding=10,
            success=True,
            logged_in=False,
        )
    ]
    agg = compute_aggregate(episodes, "random")
    path = save_aggregate_csv([agg], tmp_path / "aggregate.csv")
    assert path.exists()


def test_save_results_json(tmp_path):
    episodes = [
        EpisodeMetrics(
            seed=42,
            episode=0,
            algorithm="random",
            total_reward=10.0,
            episode_length=50,
            vulnerabilities=1,
            confirmed_vulnerabilities=0,
            endpoints_discovered=5,
            endpoint_coverage=0.5,
            steps_to_first_finding=10,
            success=True,
            logged_in=False,
        )
    ]
    agg = compute_aggregate(episodes, "random")
    path = save_results_json(episodes, [agg], {"seeds": [42]}, tmp_path / "results.json")
    assert path.exists()
    assert "aggregate" in path.read_text()
