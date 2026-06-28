"""Production resilience: retries, backoff, and circuit breaker."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    """Simple circuit breaker for failing HTTP targets."""

    failure_threshold: int = 5
    reset_timeout_seconds: float = 30.0
    failures: int = 0
    state: str = "closed"  # closed | open | half_open
    _opened_at: float = 0.0

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = "open"
            self._opened_at = time.monotonic()
            logger.warning("Circuit breaker OPEN after %s failures", self.failures)

    def allow_request(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.monotonic() - self._opened_at >= self.reset_timeout_seconds:
                self.state = "half_open"
                logger.info("Circuit breaker HALF-OPEN — probing target")
                return True
            return False
        return True  # half_open allows one probe


@dataclass
class RetryConfig:
    """Retry policy for production HTTP requests."""

    max_retries: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0


def retry_with_backoff(
    fn,
    retry_config: RetryConfig | None = None,
    circuit_breaker: CircuitBreaker | None = None,
):
    """Execute callable with exponential backoff and optional circuit breaker."""
    retry_config = retry_config or RetryConfig()
    circuit_breaker = circuit_breaker or CircuitBreaker()

    if not circuit_breaker.allow_request():
        raise RuntimeError("Circuit breaker open — target temporarily blocked")

    last_exc: Exception | None = None
    for attempt in range(retry_config.max_retries + 1):
        try:
            result = fn()
            if getattr(result, "status_code", 1) == 0:
                circuit_breaker.record_failure()
            else:
                circuit_breaker.record_success()
            return result
        except Exception as exc:
            last_exc = exc
            circuit_breaker.record_failure()
            if attempt >= retry_config.max_retries:
                break
            delay = min(
                retry_config.base_delay_seconds * (2**attempt),
                retry_config.max_delay_seconds,
            )
            logger.debug(
                "Retry %s/%s after %.1fs: %s", attempt + 1, retry_config.max_retries, delay, exc
            )
            time.sleep(delay)

    raise RuntimeError(f"Request failed after {retry_config.max_retries} retries") from last_exc
