"""Per-client rate limit on the chat endpoints (Day 7, T12)."""

from fastapi.testclient import TestClient

from customer_agent.api.rate_limit import SlidingWindowLimiter
from customer_agent.config import Settings
from customer_agent.main import create_app
from tests.unit.api.test_chat import FakeChat, FakeVerifier


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_window_allows_the_limit_then_reports_when_to_retry() -> None:
    clock = Clock()
    limiter = SlidingWindowLimiter(2, window=60, clock=clock)

    assert limiter.retry_after("a") is None
    clock.now = 10
    assert limiter.retry_after("a") is None
    clock.now = 20
    assert limiter.retry_after("a") == 40  # the oldest hit leaves the window at t=60
    assert limiter.retry_after("b") is None  # keys are independent
    clock.now = 60
    assert limiter.retry_after("a") is None


def test_idle_keys_are_pruned() -> None:
    clock = Clock()
    limiter = SlidingWindowLimiter(1, window=60, clock=clock)
    for n in range(10_001):
        limiter.retry_after(f"ip:{n}")
    clock.now = 61

    limiter.retry_after("new")

    assert len(limiter) == 1


def test_excess_messages_get_429_with_retry_after_per_identity() -> None:
    app = create_app(
        Settings(app_env="test", rate_limit_per_minute=2),
        chat=FakeChat(),  # type: ignore[arg-type]
        verifier=FakeVerifier(),
    )
    with TestClient(app) as client:
        anonymous = [client.post("/v1/chat", json={"message": "Hallo"}) for _ in range(3)]
        customer = client.post(
            "/v1/chat", json={"message": "Hallo"}, headers={"authorization": "Bearer anna"}
        )

    assert [r.status_code for r in anonymous] == [200, 200, 429]
    rejected = anonymous[2]
    assert rejected.headers["content-type"] == "application/problem+json"
    assert int(rejected.headers["retry-after"]) > 0
    assert customer.status_code == 200  # logged-in customers have their own budget


def test_zero_disables_the_limit() -> None:
    app = create_app(Settings(app_env="test", rate_limit_per_minute=0), chat=FakeChat())  # type: ignore[arg-type]
    with TestClient(app) as client:
        responses = [client.post("/v1/chat", json={"message": "Hallo"}) for _ in range(30)]

    statuses = {r.status_code for r in responses}
    assert statuses == {200}
