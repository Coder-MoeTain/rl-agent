"""Tests for seed utilities."""

from utils.seeds import get_rng, set_global_seed


def test_set_global_seed_deterministic():
    set_global_seed(42)
    r1 = get_rng(42).integers(0, 1000, size=5)
    r2 = get_rng(42).integers(0, 1000, size=5)
    assert list(r1) == list(r2)


def test_get_rng_without_seed():
    rng = get_rng()
    val = rng.integers(0, 100)
    assert 0 <= val < 100
