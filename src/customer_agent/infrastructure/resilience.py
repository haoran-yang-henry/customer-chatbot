"""Circuit breakers for external model providers (Day 7, NFR-11).

Timeouts and the SDK's retries with exponential backoff bound a single call. During a real
outage every turn would still wait for its timeouts; the breaker stops that: after
``failure_threshold`` consecutive failures it fails fast for ``reset_after`` seconds, then lets
one trial call through (half-open) and closes again on success. One breaker per endpoint, shared
by all roles that use it (router, classifier, answer).
"""

import time
from collections.abc import Callable
from enum import StrEnum

from langchain_core.embeddings import Embeddings

from customer_agent.application.ports import ProviderUnavailableError
from customer_agent.observability import get_logger
from customer_agent.observability.metrics import CIRCUIT_OPEN, PROVIDER_FAILURES

log = get_logger("resilience")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        dependency: str,
        *,
        failure_threshold: int = 3,
        reset_after: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dependency = dependency
        self._threshold = failure_threshold
        self._reset_after = reset_after
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._trial_running = False
        CIRCUIT_OPEN.labels(dependency=dependency).set(0)

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._clock() - self._opened_at >= self._reset_after:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def before_call(self) -> None:
        """Raises ``ProviderUnavailableError`` instead of calling a provider known to be down."""
        state = self.state
        if state is CircuitState.OPEN or (state is CircuitState.HALF_OPEN and self._trial_running):
            raise ProviderUnavailableError(f"{self.dependency}: circuit open")
        if state is CircuitState.HALF_OPEN:
            self._trial_running = True  # exactly one trial call; the rest keep failing fast

    def record_success(self) -> None:
        if self._opened_at is not None:
            log.info("circuit closed dependency=%s", self.dependency)
        self._failures, self._opened_at, self._trial_running = 0, None, False
        CIRCUIT_OPEN.labels(dependency=self.dependency).set(0)

    def abandon(self) -> None:
        """A call ended without a verdict (cancelled, stream closed early): free the trial."""
        self._trial_running = False

    def record_failure(self) -> None:
        PROVIDER_FAILURES.labels(dependency=self.dependency).inc()
        self._failures += 1
        self._trial_running = False
        if self._opened_at is not None or self._failures >= self._threshold:
            self._opened_at = self._clock()
            CIRCUIT_OPEN.labels(dependency=self.dependency).set(1)
            log.warning(
                "circuit open dependency=%s failures=%d for=%.0fs",
                self.dependency,
                self._failures,
                self._reset_after,
            )


class BreakerEmbeddings(Embeddings):
    """``Embeddings`` decorator with its own breaker (a different endpoint)."""

    def __init__(self, inner: Embeddings, breaker: CircuitBreaker) -> None:
        self._inner = inner
        self._breaker = breaker

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Use aembed_documents")

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("Use aembed_query")

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self._breaker.before_call()
        try:
            vectors = await self._inner.aembed_documents(texts)
        except ProviderUnavailableError:
            self._breaker.record_failure()
            raise
        except BaseException:
            self._breaker.abandon()
            raise
        self._breaker.record_success()
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        self._breaker.before_call()
        try:
            vector = await self._inner.aembed_query(text)
        except ProviderUnavailableError:
            self._breaker.record_failure()
            raise
        except BaseException:
            self._breaker.abandon()
            raise
        self._breaker.record_success()
        return vector
