"""Random and rule-based baseline agents for benchmarking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Set

import numpy as np

from gym_pentest.actions import EXPLOIT_ACTIONS, RECON_ACTIONS, ActionCategory, ACTIONS


class BaseAgent(ABC):
    """Abstract baseline agent interface."""

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        ...

    def reset(self) -> None:
        """Reset agent state between episodes."""


class RandomAgent(BaseAgent):
    """Uniform random action selection."""

    def __init__(self, num_actions: int, seed: Optional[int] = None) -> None:
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        return int(self.rng.integers(0, self.num_actions)), None


class RuleBasedAgent(BaseAgent):
    """Heuristic agent: recon first, then exploit, avoid duplicates."""

    RECON_SEQUENCE = [0, 1, 2, 5, 6, 11, 13]
    EXPLOIT_SEQUENCE = [3, 4, 7, 8, 9, 10, 14, 12]

    def __init__(self, num_actions: int, seed: Optional[int] = None) -> None:
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)
        self._used_actions: Set[int] = set()
        self._phase = "recon"

    def reset(self) -> None:
        self._used_actions.clear()
        self._phase = "recon"

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        sequence = self.RECON_SEQUENCE if self._phase == "recon" else self.EXPLOIT_SEQUENCE
        for action in sequence:
            if action not in self._used_actions and action < self.num_actions:
                self._used_actions.add(action)
                if self._phase == "recon" and action == self.RECON_SEQUENCE[-1]:
                    self._phase = "exploit"
                return action, None
        # Fallback: pick unused random action
        available = [a for a in range(self.num_actions) if a not in self._used_actions]
        if available:
            action = int(self.rng.choice(available))
            self._used_actions.add(action)
            return action, None
        action = int(self.rng.integers(0, self.num_actions))
        return action, None


def get_action_mask_for_role(role: str) -> Set[int]:
    """Return allowed actions for multi-agent role."""
    if role == "recon":
        return RECON_ACTIONS
    if role == "exploit":
        return EXPLOIT_ACTIONS | {a.id for a in ACTIONS if a.category == ActionCategory.CONFIRM}
    return set(range(len(ACTIONS)))
