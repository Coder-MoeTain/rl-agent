"""Evaluation metrics for pentesting RL experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeMetrics:
    """Metrics from a single evaluation episode."""

    seed: int
    episode: int
    algorithm: str
    total_reward: float
    episode_length: int
    vulnerabilities: int
    confirmed_vulnerabilities: int
    endpoints_discovered: int
    endpoint_coverage: float
    steps_to_first_finding: Optional[int]
    success: bool  # positive reward
    logged_in: bool
    vuln_types: List[str] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    """Aggregated metrics across episodes/seeds."""

    algorithm: str
    num_seeds: int
    episodes_per_seed: int
    mean_reward: float
    std_reward: float
    mean_endpoint_coverage: float
    mean_vuln_discovery_rate: float
    mean_steps_to_first_finding: float
    success_rate: float
    mean_episode_length: float
    training_time_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_aggregate(episodes: List[EpisodeMetrics], algorithm: str, training_time: Optional[float] = None) -> AggregateMetrics:
    """Compute aggregate statistics from episode metrics."""
    import numpy as np

    rewards = [e.total_reward for e in episodes]
    coverages = [e.endpoint_coverage for e in episodes]
    vuln_rates = [e.vulnerabilities for e in episodes]
    successes = [e.success for e in episodes]
    lengths = [e.episode_length for e in episodes]

    steps_to_find = [e.steps_to_first_finding for e in episodes if e.steps_to_first_finding is not None]
    mean_steps = float(np.mean(steps_to_find)) if steps_to_find else float("inf")

    seeds = set(e.seed for e in episodes)

    return AggregateMetrics(
        algorithm=algorithm,
        num_seeds=len(seeds),
        episodes_per_seed=len(episodes) // max(len(seeds), 1),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_endpoint_coverage=float(np.mean(coverages)),
        mean_vuln_discovery_rate=float(np.mean(vuln_rates)),
        mean_steps_to_first_finding=mean_steps,
        success_rate=sum(successes) / len(successes) if successes else 0.0,
        mean_episode_length=float(np.mean(lengths)),
        training_time_seconds=training_time,
    )
