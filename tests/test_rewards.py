"""Tests for reward logic."""


def test_step_penalty_applied(mock_env):
    mock_env.reset(seed=0)
    _, reward, _, _, _ = mock_env.step(9)
    assert reward < 1.0


def test_login_success_reward(mock_env):
    mock_env.reset(seed=0)
    _, reward, _, _, info = mock_env.step(3)
    assert reward > 5.0
    assert info["logged_in"] is True


def test_xss_reward_on_reflection(mock_env):
    mock_env.reset(seed=0)
    _, reward, _, _, info = mock_env.step(4)
    assert reward > 20.0
    assert info["vulnerabilities"] >= 1


def test_duplicate_action_penalty(mock_env):
    mock_env.reset(seed=0)
    _, r1, _, _, _ = mock_env.step(0)
    _, r2, _, _, info = mock_env.step(0)
    assert info["duplicate_actions"] >= 1
    assert r2 < r1


def test_idor_reward(mock_env):
    mock_env.reset(seed=0)
    _, reward, _, _, info = mock_env.step(10)
    assert reward > 10.0
    assert info["vulnerabilities"] >= 1
