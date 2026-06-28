"""Evaluation metrics for defensive vulnerability assessment experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
    steps_to_first_finding: int | None
    success: bool
    logged_in: bool
    forms_found: int = 0
    params_found: int = 0
    duplicate_findings: int = 0
    report_generated: bool = False
    requests_made: int = 0
    challenges_solved: int = 0
    scoreboard_progress: float = 0.0
    steps_to_first_challenge: int | None = None
    vuln_types: list[str] = field(default_factory=list)
    ablation: str | None = None

    @property
    def confirmed_finding_rate(self) -> float:
        return self.confirmed_vulnerabilities / max(self.vulnerabilities, 1)

    @property
    def requests_per_confirmed_finding(self) -> float:
        if self.confirmed_vulnerabilities == 0:
            return float("inf")
        return self.requests_made / self.confirmed_vulnerabilities


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
    mean_confirmed_finding_rate: float
    mean_steps_to_first_finding: float
    mean_requests_per_confirmed_finding: float
    mean_form_discovery: float
    mean_param_discovery: float
    success_rate: float
    mean_episode_length: float
    mean_challenges_solved: float = 0.0
    mean_scoreboard_progress: float = 0.0
    training_time_seconds: float | None = None
    ablation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_aggregate(
    episodes: list[EpisodeMetrics],
    algorithm: str,
    training_time: float | None = None,
    ablation: str | None = None,
) -> AggregateMetrics:
    """Compute aggregate statistics from episode metrics."""
    import numpy as np

    rewards = [e.total_reward for e in episodes]
    coverages = [e.endpoint_coverage for e in episodes]
    vuln_rates = [e.vulnerabilities for e in episodes]
    confirmed_rates = [e.confirmed_finding_rate for e in episodes]
    successes = [e.success for e in episodes]
    lengths = [e.episode_length for e in episodes]
    challenges = [e.challenges_solved for e in episodes]
    progress = [e.scoreboard_progress for e in episodes]
    forms = [e.forms_found for e in episodes]
    params = [e.params_found for e in episodes]
    req_per_confirmed = [
        e.requests_per_confirmed_finding
        for e in episodes
        if e.confirmed_vulnerabilities > 0
    ]

    steps_to_find = [
        e.steps_to_first_finding for e in episodes if e.steps_to_first_finding is not None
    ]
    mean_steps = float(np.mean(steps_to_find)) if steps_to_find else float("inf")
    mean_req = float(np.mean(req_per_confirmed)) if req_per_confirmed else float("inf")

    seeds = set(e.seed for e in episodes)

    return AggregateMetrics(
        algorithm=algorithm,
        num_seeds=len(seeds),
        episodes_per_seed=len(episodes) // max(len(seeds), 1),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_endpoint_coverage=float(np.mean(coverages)),
        mean_vuln_discovery_rate=float(np.mean(vuln_rates)),
        mean_confirmed_finding_rate=float(np.mean(confirmed_rates)),
        mean_steps_to_first_finding=mean_steps,
        mean_requests_per_confirmed_finding=mean_req,
        mean_form_discovery=float(np.mean(forms)),
        mean_param_discovery=float(np.mean(params)),
        success_rate=sum(successes) / len(successes) if successes else 0.0,
        mean_episode_length=float(np.mean(lengths)),
        mean_challenges_solved=float(np.mean(challenges)),
        mean_scoreboard_progress=float(np.mean(progress)),
        training_time_seconds=training_time,
        ablation=ablation,
    )
