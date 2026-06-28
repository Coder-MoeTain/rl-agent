"""Tests for action validity and execution."""

from gym_pentest.actions import ACTION_NAMES, NUM_ACTIONS, RECON_ACTIONS, TEST_ACTIONS


def test_num_actions(mock_env):
    assert mock_env.action_space.n == NUM_ACTIONS
    assert NUM_ACTIONS == 16


def test_all_actions_executable(mock_env):
    mock_env.reset(seed=0)
    for action in range(NUM_ACTIONS):
        obs, reward, terminated, truncated, info = mock_env.step(action)
        assert obs is not None
        assert isinstance(reward, float)
        assert info["last_action"] == ACTION_NAMES[action]
        if terminated or truncated:
            mock_env.reset(seed=action)


def test_recon_actions_subset():
    assert 0 in RECON_ACTIONS
    assert 13 in RECON_ACTIONS


def test_test_actions_subset():
    assert 3 in TEST_ACTIONS
    assert 4 in TEST_ACTIONS


def test_generate_report_action(mock_env):
    mock_env.reset(seed=0)
    _, reward, _, _, info = mock_env.step(15)
    assert info["report_generated"] is True
    assert reward > 0
