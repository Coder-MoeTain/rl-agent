"""Tests for environment reward calculation and observation."""


def test_calculate_reward_duplicate_penalty(mock_env):
    mock_env.reset(seed=0)
    mock_env._calculate_reward(0, 1.0, False)  # first action
    reward = mock_env._calculate_reward(0, 1.0, False)  # duplicate
    assert reward < 1.0 + mock_env.reward_config.step_penalty


def test_get_observation_shape(mock_env):
    obs = mock_env._get_observation()
    assert obs.shape == (128,)


def test_update_graph_from_links(mock_env):
    mock_env.reset(seed=0)
    new = mock_env._update_graph_from_links("http://localhost:3000/", {"/login", "/products"})
    assert new >= 2
    assert mock_env.attack_graph.number_of_nodes() >= 2


def test_log_step_no_error(mock_env, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    mock_env.reset(seed=0)
    mock_env.step(0)
    assert mock_env.steps == 1
