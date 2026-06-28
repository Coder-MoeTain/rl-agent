"""Tests for baseline agents."""

import numpy as np

from agents.baselines import RandomAgent, RuleBasedAgent


def test_random_agent_range():
    agent = RandomAgent(16, seed=42)
    for _ in range(20):
        action, _ = agent.predict(np.zeros(128))
        assert 0 <= action < 16


def test_rule_based_recon_first():
    agent = RuleBasedAgent(16, seed=42)
    agent.reset()
    action, _ = agent.predict(np.zeros(128))
    assert action == 0
