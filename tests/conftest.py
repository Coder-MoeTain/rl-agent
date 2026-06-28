"""Pytest fixtures for pentesting environment tests."""

from __future__ import annotations

from typing import Dict, Tuple

import pytest

from gym_pentest.http_client import HttpResponse, MockHttpClient
from gym_pentest.env import PentestEnv


def _make_mock_handlers() -> Dict[Tuple[str, str], HttpResponse]:
    """Build mock HTTP responses for lab endpoints."""
    return {
        ("GET", "/"): HttpResponse(
            url="http://localhost:3000/",
            status_code=200,
            text='<html><a href="/#/login">Login</a><a href="/rest/products">Products</a></html>',
            headers={},
        ),
        ("GET", "/#/login"): HttpResponse(
            url="http://localhost:3000/#/login",
            status_code=200,
            text='<html><form action="/rest/user/login"><input name="email"/><input name="password"/></form></html>',
            headers={},
        ),
        ("GET", "/rest/feedback"): HttpResponse(
            url="http://localhost:3000/rest/feedback",
            status_code=200,
            text='{"feedback": []}',
            headers={},
        ),
        ("GET", "/rest/products"): HttpResponse(
            url="http://localhost:3000/rest/products",
            status_code=200,
            text='{"data": [{"id": 1, "name": "Apple"}]}',
            headers={},
        ),
        ("GET", "/rest/user/whoami"): HttpResponse(
            url="http://localhost:3000/rest/user/whoami",
            status_code=200,
            text='{"user": {}}',
            headers={},
        ),
        ("GET", "/rest/admin/application-configuration"): HttpResponse(
            url="http://localhost:3000/rest/admin/application-configuration",
            status_code=200,
            text='{"config": {"apiKey": "secret123"}}',
            headers={},
        ),
        ("GET", "/api/Users/1"): HttpResponse(
            url="http://localhost:3000/api/Users/1",
            status_code=200,
            text='{"id": 1, "email": "admin@juice-sh.op", "password": "hash"}',
            headers={},
        ),
        ("GET", "/rest/basket/1"): HttpResponse(
            url="http://localhost:3000/rest/basket/1",
            status_code=200,
            text='{"data": {"items": [{"Product": "Apple"}]}}',
            headers={},
        ),
    }


def _post_handler(method: str, path: str, json_data: dict | None) -> HttpResponse:
    """Dynamic POST handler for login, feedback, search."""
    if path == "/rest/user/login":
        email = (json_data or {}).get("email", "")
        if "' OR 1=1--" in email:
            return HttpResponse(
                url="http://localhost:3000/rest/user/login",
                status_code=500,
                text='{"error": "SQLITE_ERROR: syntax error"}',
                headers={},
            )
        if email == "attacker@example.com":
            return HttpResponse(
                url="http://localhost:3000/rest/user/login",
                status_code=200,
                text='{"authentication": {"token": "abc123"}}',
                headers={},
            )
        return HttpResponse(
            url="http://localhost:3000/rest/user/login",
            status_code=401,
            text='{"error": "Invalid credentials"}',
            headers={},
        )

    if path == "/rest/feedback":
        comment = (json_data or {}).get("comment", "")
        return HttpResponse(
            url="http://localhost:3000/rest/feedback",
            status_code=201,
            text=f'{{"comment": "{comment}", "status": "ok"}}',
            headers={},
        )

    if path == "/rest/products/search":
        return HttpResponse(
            url="http://localhost:3000/rest/products/search",
            status_code=200,
            text='{"products": []}',
            headers={},
        )

    return HttpResponse(url=f"http://localhost:3000{path}", status_code=404, text="", headers={})


@pytest.fixture
def mock_http_client() -> MockHttpClient:
    return build_mock_http_client()


def build_mock_http_client() -> MockHttpClient:
    """Build mock HTTP client (usable outside pytest fixtures)."""
    handlers = _make_mock_handlers()
    client = MockHttpClient(handlers=handlers)

    # Wrap POST dynamically
    original_lookup = client._lookup

    def lookup(method, base_url, path, json_data=None):
        if method.upper() == "POST":
            return _post_handler(method, path, json_data)
        return original_lookup(method, base_url, path, json_data)

    client._lookup = lookup  # type: ignore[method-assign]
    return client


@pytest.fixture
def mock_env(mock_http_client: MockHttpClient) -> PentestEnv:
    return PentestEnv(
        base_url="http://localhost:3000",
        max_steps=50,
        http_client=mock_http_client,
        disable_scope_guard=True,
        mission_vuln_threshold=1,
    )
