"""Quick evaluation test using mocked HTTP (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from agents.baselines import RandomAgent, RuleBasedAgent
from gym_pentest.env import PentestEnv
from gym_pentest.scoreboard import MockScoreboard
from gym_pentest.mock_http import build_mock_http_client


def test_offline_eval_random():
    env = PentestEnv(
        http_client=build_mock_http_client(),
        scoreboard=MockScoreboard(),
        disable_scope_guard=True,
        max_steps=20,
    )
    agent = RandomAgent(env.action_space.n, seed=42)
    obs, _ = env.reset(seed=42)
    total = 0.0
    for _ in range(20):
        action, _ = agent.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total += reward
        if done or truncated:
            break
    assert info["steps"] >= 1


def test_offline_eval_rule_based():
    env = PentestEnv(
        http_client=build_mock_http_client(),
        scoreboard=MockScoreboard(),
        disable_scope_guard=True,
        max_steps=20,
    )
    agent = RuleBasedAgent(env.action_space.n, seed=42)
    agent.reset()
    obs, _ = env.reset(seed=42)
    for _ in range(10):
        action, _ = agent.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    assert info["discovered_count"] >= 1
