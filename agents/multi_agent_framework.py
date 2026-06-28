"""Multi-agent coordination framework for defensive vulnerability assessment.

Five specialized agents cooperate via shared memory (discovery graph + evidence store):
  1. ReconAgent       — endpoint/form/parameter discovery
  2. TestingAgent     — controlled vulnerability probes in lab targets
  3. EvidenceAgent    — validation and deduplication of findings
  4. RiskAgent        — OWASP mapping and severity scoring
  5. ReportAgent      — remediation report generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from gym_pentest.actions import (
    CONFIRM_ACTIONS,
    NUM_ACTIONS,
    RECON_ACTIONS,
    REPORT_ACTIONS,
    TEST_ACTIONS,
    get_action_name,
)
from gym_pentest.report import generate_finding_reports

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Specialized agent roles in the multi-agent assessment framework."""

    RECON = "recon"
    TESTING = "testing"
    EVIDENCE = "evidence"
    RISK = "risk"
    REPORT = "report"


ROLE_ACTION_MASKS: dict[AgentRole, set[int]] = {
    AgentRole.RECON: RECON_ACTIONS,
    AgentRole.TESTING: TEST_ACTIONS,
    AgentRole.EVIDENCE: CONFIRM_ACTIONS,
    AgentRole.RISK: CONFIRM_ACTIONS | RECON_ACTIONS,
    AgentRole.REPORT: REPORT_ACTIONS,
}


@dataclass
class SharedMemory:
    """Shared state accessible to all agents during an episode."""

    discovered_endpoints: set[str] = field(default_factory=set)
    forms_found: int = 0
    params_found: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confirmed_count: int = 0
    risk_scores: dict[str, float] = field(default_factory=dict)
    reports: list[dict[str, Any]] = field(default_factory=list)
    graph_nodes: int = 0

    def update_from_info(self, info: dict[str, Any]) -> None:
        """Synchronize shared memory from environment info dict."""
        self.forms_found = info.get("forms_found", 0)
        self.params_found = info.get("params_found", 0)
        self.evidence = info.get("evidence", [])
        self.confirmed_count = info.get("confirmed_vulnerabilities", 0)
        self.graph_nodes = info.get("graph_nodes", 0)
        self.reports = info.get("finding_reports", [])


@dataclass
class AgentMetrics:
    """Per-agent evaluation metrics."""

    role: str
    actions_taken: int = 0
    successful_actions: int = 0
    discoveries: int = 0
    findings_validated: int = 0
    reports_generated: int = 0

    @property
    def success_rate(self) -> float:
        return self.successful_actions / max(self.actions_taken, 1)


class MultiAgentCoordinator:
    """Round-robin coordinator with role-specific action masks and shared memory."""

    ROLE_SEQUENCE = [
        AgentRole.RECON,
        AgentRole.TESTING,
        AgentRole.EVIDENCE,
        AgentRole.RISK,
        AgentRole.REPORT,
    ]

    def __init__(self, num_actions: int = NUM_ACTIONS, seed: int | None = None) -> None:
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)
        self.shared_memory = SharedMemory()
        self._role_index = 0
        self._metrics: dict[AgentRole, AgentMetrics] = {
            role: AgentMetrics(role=role.value) for role in AgentRole
        }

    def reset(self) -> None:
        self.shared_memory = SharedMemory()
        self._role_index = 0
        self._metrics = {role: AgentMetrics(role=role.value) for role in AgentRole}

    @property
    def current_role(self) -> AgentRole:
        return self.ROLE_SEQUENCE[self._role_index % len(self.ROLE_SEQUENCE)]

    def advance_role(self) -> AgentRole:
        self._role_index += 1
        return self.current_role

    def get_action_mask(self, role: AgentRole | None = None) -> set[int]:
        role = role or self.current_role
        return ROLE_ACTION_MASKS.get(role, set(range(self.num_actions)))

    def select_action(
        self,
        obs: np.ndarray,
        allowed: set[int] | None = None,
        deterministic: bool = False,
    ) -> tuple[int, AgentRole]:
        """Select action for current role from allowed mask."""
        role = self.current_role
        allowed = allowed or self.get_action_mask(role)
        if not allowed:
            allowed = set(range(self.num_actions))
        action = int(self.rng.choice(list(allowed)))
        self._metrics[role].actions_taken += 1
        logger.debug("Agent %s selected action %s", role.value, get_action_name(action))
        return action, role

    def observe_step(self, role: AgentRole, reward: float, info: dict[str, Any]) -> None:
        """Update shared memory and agent metrics after environment step."""
        self.shared_memory.update_from_info(info)
        metrics = self._metrics[role]
        if reward > 0:
            metrics.successful_actions += 1
        if info.get("discovered_count", 0) > len(self.shared_memory.discovered_endpoints):
            metrics.discoveries += 1
        if info.get("confirmed_vulnerabilities", 0) > self.shared_memory.confirmed_count:
            metrics.findings_validated += 1
        if info.get("report_generated"):
            metrics.reports_generated += 1
        self.advance_role()

    def get_metrics(self) -> dict[str, dict[str, Any]]:
        return {
            role.value: {
                "actions_taken": m.actions_taken,
                "successful_actions": m.successful_actions,
                "success_rate": m.success_rate,
                "discoveries": m.discoveries,
                "findings_validated": m.findings_validated,
                "reports_generated": m.reports_generated,
            }
            for role, m in self._metrics.items()
        }

    def assess_risk(self) -> list[dict[str, Any]]:
        """Risk assessment agent: map evidence to OWASP categories and scores."""
        reports = generate_finding_reports(self.shared_memory.evidence)
        for report in reports:
            self.shared_memory.risk_scores[report.finding_id] = report.risk_score
        return [r.to_dict() for r in reports]


class MultiAgentRLAgent:
    """Wrapper exposing multi-agent coordinator as a predict()-compatible agent."""

    def __init__(self, seed: int | None = None) -> None:
        self.coordinator = MultiAgentCoordinator(seed=seed)

    def reset(self) -> None:
        self.coordinator.reset()

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        action, _ = self.coordinator.select_action(obs, deterministic=deterministic)
        return action, None

    @property
    def current_role(self) -> AgentRole:
        return self.coordinator.current_role
