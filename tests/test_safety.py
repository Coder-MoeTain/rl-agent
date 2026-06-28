"""Tests for safety controls."""

import pytest

from gym_pentest.safety import (
    ActionLogger,
    EmergencyStop,
    RateLimiter,
    SafetyConfig,
    SafetyController,
)
from gym_pentest.scope import ScopeGuard


def test_rate_limiter_counts_requests():
    limiter = RateLimiter(requests_per_second=1000.0)
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    assert limiter.request_count == 2


def test_emergency_stop_triggers_on_scope_violations():
    stop = EmergencyStop(max_scope_violations=2)
    stop.record_scope_violation()
    assert not stop.triggered
    stop.record_scope_violation()
    assert stop.triggered
    assert stop.reason is not None


def test_safety_controller_blocks_after_max_requests():
    config = SafetyConfig(max_requests_per_episode=2, safe_mode=False)
    controller = SafetyController(config)
    assert controller.before_request() is True
    assert controller.before_request() is True
    assert controller.before_request() is False
    assert controller.emergency_stop.triggered


def test_action_logger_records_entries():
    logger = ActionLogger()
    logger.log(1, 0, "crawl_page", 1.0, {"discovered_count": 1, "vulnerabilities": 0})
    assert logger.count == 1
    assert logger.to_list()[0]["action_name"] == "crawl_page"


def test_scope_blocks_public_internet():
    guard = ScopeGuard("http://localhost:3000", safety_config=SafetyConfig())
    assert guard.is_in_scope("http://evil.com/attack") is False


def test_scope_allows_localhost():
    guard = ScopeGuard("http://localhost:3000")
    assert guard.is_in_scope("http://localhost:3000/rest/products") is True


def test_validate_raises_for_out_of_scope():
    guard = ScopeGuard("http://localhost:3000")
    with pytest.raises(ValueError, match="out of authorized scope"):
        guard.validate_or_raise("http://evil.com/attack")
