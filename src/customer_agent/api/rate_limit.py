"""Per-client rate limit for the chat endpoints (Day 7, threat model T12), and the daily cap of
all chat messages together (the public demo).

A sliding window in process memory: enough for one instance and for measuring the cost of the
check. Several instances need a shared store (Redis); the dependency keeps that a local change.
Behind a reverse proxy, run uvicorn with ``--proxy-headers`` so the client address is real. The
daily count is in the database: it survives a restart, which a scaled-to-zero service has often.
"""

import math
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from customer_agent.api.auth import CurrentPrincipal
from customer_agent.application.ports import UsageCounter
from customer_agent.observability.metrics import RATE_LIMITED

# The frontend tells this 429 apart from the per-minute one by its detail.
DAILY_LIMIT_DETAIL = "the demo's daily message limit is reached; please try again tomorrow"

_PRUNE_ABOVE = 10_000  # keys kept before idle ones are dropped


class SlidingWindowLimiter:
    """At most ``limit`` requests per ``window`` seconds per key."""

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._limit = limit
        self._window = window
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def retry_after(self, key: str) -> float | None:
        """``None`` if the request may pass (and is counted), else seconds until it may."""
        now = self._clock()
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self._window:
            hits.popleft()
        if len(hits) >= self._limit:
            return hits[0] + self._window - now
        hits.append(now)
        if len(self._hits) > _PRUNE_ABOVE:
            self._prune(now)
        return None

    def _prune(self, now: float) -> None:
        idle = [k for k, h in self._hits.items() if not h or h[-1] <= now - self._window]
        for key in idle:
            del self._hits[key]

    def __len__(self) -> int:
        return len(self._hits)


async def enforce_rate_limit(request: Request, principal: CurrentPrincipal) -> None:
    """Customers are limited by account, anonymous clients by address."""
    limiter: SlidingWindowLimiter | None = getattr(request.app.state, "limiter", None)
    if limiter is None:
        return
    if principal is not None:
        scope, key = "customer", f"customer:{principal.customer_id}"
    else:
        scope, key = "ip", f"ip:{request.client.host if request.client else '-'}"
    retry = limiter.retry_after(key)
    if retry is not None:
        RATE_LIMITED.labels(scope=scope).inc()
        raise HTTPException(
            429,
            "too many messages in a short time; please wait a moment",
            {"Retry-After": str(math.ceil(retry))},
        )


RateLimited = Annotated[None, Depends(enforce_rate_limit)]


async def enforce_message_limits(
    request: Request, principal: CurrentPrincipal, _: RateLimited
) -> None:
    """The per-client limit first, so a client already over it does not use up the day. Exempt
    accounts (the owner's) still have the per-minute limit."""
    limit: int = getattr(request.app.state, "daily_limit", 0)
    usage: UsageCounter | None = getattr(request.app.state, "usage", None)
    exempt: frozenset[str] = getattr(request.app.state, "daily_exempt", frozenset())
    if not limit or usage is None:
        return
    if principal is not None and principal.username in exempt:
        return
    now = datetime.now(UTC)
    if await usage.count_message(now.date()) <= limit:
        return
    RATE_LIMITED.labels(scope="daily").inc()
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    raise HTTPException(
        429, DAILY_LIMIT_DETAIL, {"Retry-After": str(math.ceil((midnight - now).total_seconds()))}
    )


MessageLimited = Annotated[None, Depends(enforce_message_limits)]
