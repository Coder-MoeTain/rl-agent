"""Result persistence utilities for experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.metrics import AggregateMetrics, EpisodeMetrics


def save_episodes_csv(episodes: list[EpisodeMetrics], output_path: Path) -> Path:
    """Save episode-level metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([e.__dict__ for e in episodes])
    df.to_csv(output_path, index=False)
    return output_path


def save_aggregate_csv(aggregates: list[AggregateMetrics], output_path: Path) -> Path:
    """Save aggregate metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([a.to_dict() for a in aggregates])
    df.to_csv(output_path, index=False)
    return output_path


def save_results_json(
    episodes: list[EpisodeMetrics],
    aggregates: list[AggregateMetrics],
    config_meta: dict[str, Any],
    output_path: Path,
) -> Path:
    """Save full experiment results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config_meta,
        "aggregate": [a.to_dict() for a in aggregates],
        "episodes": [e.__dict__ for e in episodes],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return output_path
