"""Tests for multi-agent framework."""

from agents.multi_agent_framework import AgentRole, MultiAgentCoordinator, MultiAgentRLAgent
from gym_pentest.actions import NUM_ACTIONS, RECON_ACTIONS, REPORT_ACTIONS, TEST_ACTIONS


def test_coordinator_role_rotation():
    coord = MultiAgentCoordinator(seed=42)
    assert coord.current_role == AgentRole.RECON
    coord.advance_role()
    assert coord.current_role == AgentRole.TESTING


def test_recon_action_mask():
    coord = MultiAgentCoordinator()
    mask = coord.get_action_mask(AgentRole.RECON)
    assert mask == RECON_ACTIONS


def test_report_action_mask():
    coord = MultiAgentCoordinator()
    mask = coord.get_action_mask(AgentRole.REPORT)
    assert mask == REPORT_ACTIONS


def test_multi_agent_rl_agent_predict():
    agent = MultiAgentRLAgent(seed=42)
    obs = __import__("numpy").zeros(128, dtype=__import__("numpy").float32)
    action, _ = agent.predict(obs)
    assert 0 <= action < NUM_ACTIONS


def test_shared_memory_update():
    coord = MultiAgentCoordinator()
    coord.shared_memory.update_from_info(
        {"forms_found": 3, "params_found": 5, "evidence": [], "graph_nodes": 10}
    )
    assert coord.shared_memory.forms_found == 3
    assert coord.shared_memory.params_found == 5
