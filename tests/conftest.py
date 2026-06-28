"""Pytest fixtures for pentesting environment tests."""

from __future__ import annotations

import pytest

from gym_pentest.env import PentestEnv
from gym_pentest.http_client import MockHttpClient
from gym_pentest.mock_http import build_mock_http_client
from gym_pentest.scoreboard import MockScoreboard


@pytest.fixture
def mock_http_client() -> MockHttpClient:
    return build_mock_http_client()


@pytest.fixture
def mock_env(mock_http_client: MockHttpClient) -> PentestEnv:
    return PentestEnv(
        base_url="http://localhost:3000",
        max_steps=50,
        http_client=mock_http_client,
        scoreboard=MockScoreboard(),
        disable_scope_guard=True,
        disable_safety_controls=True,
        mission_vuln_threshold=1,
        mission_challenge_threshold=1,
    )
