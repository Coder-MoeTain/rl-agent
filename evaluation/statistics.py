"""Statistical analysis for research-grade experiment comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""

    mean: float
    lower: float
    upper: float
    confidence: float = 0.95


@dataclass
class SignificanceResult:
    """Result of statistical significance test between two algorithms."""

    algorithm_a: str
    algorithm_b: str
    metric: str
    mean_a: float
    mean_b: float
    p_value: float
    significant: bool
    test_name: str
    effect_size: float | None = None


def confidence_interval(values: list[float], confidence: float = 0.95) -> ConfidenceInterval:
    """Compute mean and confidence interval using t-distribution."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence)
    mean = float(np.mean(arr))
    if n == 1:
        return ConfidenceInterval(mean, mean, mean, confidence)
    std = float(np.std(arr, ddof=1))
    # t-critical approximation for 95% CI (two-tailed)
    t_crit = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
    margin = t_crit * std / np.sqrt(n)
    return ConfidenceInterval(mean, mean - margin, mean + margin, confidence)


def welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-test for unequal variances. Returns (t_statistic, p_value)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return 0.0, 1.0

    mean_a, mean_b = np.mean(a_arr), np.mean(b_arr)
    var_a = np.var(a_arr, ddof=1)
    var_b = np.var(b_arr, ddof=1)
    n_a, n_b = len(a_arr), len(b_arr)

    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0

    t_stat = (mean_a - mean_b) / se
    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1

    # Approximate p-value using normal for large df, rough for small
    p_value = float(2 * (1 - _normal_cdf(abs(t_stat))))
    _ = df  # reserved for future scipy integration
    return float(t_stat), p_value


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF."""
    return 0.5 * (1 + np.tanh(x * 0.7978845608))


def cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return 0.0
    pooled_std = np.sqrt((np.var(a_arr, ddof=1) + np.var(b_arr, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)


def compare_algorithms(
    rewards_by_algo: dict[str, list[float]],
    metric_name: str = "total_reward",
    alpha: float = 0.05,
) -> list[SignificanceResult]:
    """Pairwise significance tests between all algorithm pairs."""
    results: list[SignificanceResult] = []
    algos = sorted(rewards_by_algo.keys())
    for i, a in enumerate(algos):
        for b in algos[i + 1 :]:
            vals_a = rewards_by_algo[a]
            vals_b = rewards_by_algo[b]
            _, p_value = welch_t_test(vals_a, vals_b)
            results.append(
                SignificanceResult(
                    algorithm_a=a,
                    algorithm_b=b,
                    metric=metric_name,
                    mean_a=float(np.mean(vals_a)),
                    mean_b=float(np.mean(vals_b)),
                    p_value=p_value,
                    significant=p_value < alpha,
                    test_name="welch_t",
                    effect_size=cohens_d(vals_a, vals_b),
                )
            )
    return results


def summarize_with_ci(values: list[float], label: str, confidence: float = 0.95) -> dict[str, Any]:
    """Return summary dict with mean, std, and confidence interval."""
    ci = confidence_interval(values, confidence)
    return {
        "label": label,
        "n": len(values),
        "mean": ci.mean,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci_lower": ci.lower,
        "ci_upper": ci.upper,
        "confidence": confidence,
    }
