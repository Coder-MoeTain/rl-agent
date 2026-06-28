"""Tests for PentestEnv reset and step mechanics."""

import numpy as np

from gym_pentest.features import OBS_DIM


def test_reset_returns_valid_observation(mock_env):
    obs, info = mock_env.reset(seed=42)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0) and np.all(obs <= 1)
    assert info["discovered_count"] == 0
    assert info["vulnerabilities"] == 0


def test_reset_is_deterministic_with_seed(mock_env):
    obs1, _ = mock_env.reset(seed=123)
    obs2, _ = mock_env.reset(seed=123)
    np.testing.assert_array_equal(obs1, obs2)


def test_step_increments_steps(mock_env):
    mock_env.reset(seed=0)
    _, _, _, _, info = mock_env.step(0)
    assert info["steps"] == 1
    _, _, _, _, info = mock_env.step(1)
    assert info["steps"] == 2


def test_step_returns_five_tuple(mock_env):
    mock_env.reset(seed=0)
    result = mock_env.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_truncation_at_max_steps(mock_env):
    mock_env.reset(seed=0)
    terminated = False
    truncated = False
    for _ in range(mock_env.max_steps):
        _, _, terminated, truncated, _ = mock_env.step(0)
    assert truncated is True


def test_crawl_discovers_endpoints(mock_env):
    mock_env.reset(seed=0)
    _, reward, _, _, info = mock_env.step(0)
    assert reward > 0
    assert info["discovered_count"] >= 1
