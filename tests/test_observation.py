"""Tests for observation shape and feature extraction."""

import numpy as np

from gym_pentest.features import OBS_DIM


def test_observation_shape(mock_env):
    obs, _ = mock_env.reset()
    assert obs.shape == (OBS_DIM,)


def test_observation_bounds(mock_env):
    mock_env.reset(seed=0)
    for action in range(min(5, mock_env.action_space.n)):
        obs, _, _, _, _ = mock_env.step(action)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)


def test_observation_changes_after_action(mock_env):
    obs_before, _ = mock_env.reset(seed=0)
    obs_after, _, _, _, _ = mock_env.step(0)
    assert not np.array_equal(obs_before, obs_after)
