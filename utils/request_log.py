"""HTTP request logging for vulnerability assessment audit trail."""

from __future__ import annotations

import logging

logger = logging.getLogger("pentest.requests")


def log_request(method: str, url: str, status_code: int, response_len: int) -> None:
    """Log an HTTP request and response summary."""
    logger.debug(
        "%s %s -> status=%s len=%s",
        method.upper(),
        url,
        status_code,
        response_len,
    )


def log_scope_block(url: str, reason: str | None = None) -> None:
    """Log a blocked out-of-scope request."""
    logger.warning("Scope block: %s (%s)", url, reason or "out of scope")
