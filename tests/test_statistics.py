"""Tests for statistical evaluation utilities."""

from evaluation.statistics import (
    cohens_d,
    compare_algorithms,
    confidence_interval,
    summarize_with_ci,
    welch_t_test,
)


def test_confidence_interval():
    values = [10.0, 12.0, 11.0, 13.0, 9.0]
    ci = confidence_interval(values)
    assert ci.lower <= ci.mean <= ci.upper


def test_welch_t_test_identical():
    a = [5.0, 5.0, 5.0, 5.0]
    b = [5.0, 5.0, 5.0, 5.0]
    _, p = welch_t_test(a, b)
    assert p == 1.0


def test_welch_t_test_different():
    a = [20.0, 22.0, 21.0, 23.0]
    b = [1.0, 2.0, 1.5, 2.5]
    _, p = welch_t_test(a, b)
    assert p < 0.05


def test_cohens_d():
    a = [10.0, 12.0, 11.0]
    b = [1.0, 2.0, 1.5]
    d = cohens_d(a, b)
    assert d > 1.0


def test_compare_algorithms():
    rewards = {"random": [1.0, 2.0, 1.5], "rule_based": [10.0, 12.0, 11.0]}
    results = compare_algorithms(rewards)
    assert len(results) == 1
    assert results[0].significant is True


def test_summarize_with_ci():
    summary = summarize_with_ci([1.0, 2.0, 3.0], "test")
    assert summary["mean"] == 2.0
    assert summary["n"] == 3
