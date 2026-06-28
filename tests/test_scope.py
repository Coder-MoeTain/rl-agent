"""Tests for scope guard."""

import pytest

from gym_pentest.scope import ScopeGuard


def test_localhost_in_scope():
    guard = ScopeGuard("http://localhost:3000")
    assert guard.is_in_scope("http://localhost:3000/rest/products") is True


def test_external_host_blocked():
    guard = ScopeGuard("http://localhost:3000")
    assert guard.is_in_scope("http://evil.com/attack") is False


def test_validate_raises_for_out_of_scope():
    guard = ScopeGuard("http://localhost:3000")
    with pytest.raises(ValueError, match="out of authorized scope"):
        guard.validate_or_raise("http://evil.com/attack")
