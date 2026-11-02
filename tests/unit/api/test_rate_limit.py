"""Per-client rate limit on the chat endpoints (Day 7, T12), and the daily cap of the demo."""

from datetime import date

from fastapi.testclient import TestClient

from customer_agent.api.rate_limit import DAILY_LIMIT_DETAIL, SlidingWindowLimiter
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


class FakeUsage:
    def __init__(self) -> None:
        self.days: dict[date, int] = {}

    async def count_message(self, day: date) -> int:
        self.days[day] = self.days.get(day, 0) + 1
        return self.days[day]


def test_messages_past_the_daily_limit_get_429_until_midnight() -> None:
    settings = Settings(app_env="test", rate_limit_per_minute=0, daily_message_limit=2)
    app = create_app(settings, chat=FakeChat(), usage=FakeUsage())  # type: ignore[arg-type]
    with TestClient(app) as client:
        responses = [client.post("/v1/chat", json={"message": "Hallo"}) for _ in range(3)]
        stream = client.post("/v1/chat/stream", json={"message": "Hallo"})

    assert [r.status_code for r in responses] == [200, 200, 429]
    rejected = responses[2]
    assert rejected.json()["detail"] == DAILY_LIMIT_DETAIL  # the frontend's cue
    assert 0 < int(rejected.headers["retry-after"]) <= 24 * 60 * 60
    assert stream.status_code == 429  # streamed messages count against the same day


def test_messages_over_the_per_minute_limit_do_not_use_up_the_day() -> None:
    usage = FakeUsage()
    settings = Settings(app_env="test", rate_limit_per_minute=1, daily_message_limit=5)
    app = create_app(settings, chat=FakeChat(), usage=usage)  # type: ignore[arg-type]
    with TestClient(app) as client:
        statuses = [
            client.post("/v1/chat", json={"message": "Hallo"}).status_code for _ in range(3)
        ]

    assert statuses == [200, 429, 429]
    assert list(usage.days.values()) == [1]


def test_without_a_daily_limit_nothing_is_counted() -> None:
    usage = FakeUsage()
    app = create_app(Settings(app_env="test"), chat=FakeChat(), usage=usage)  # type: ignore[arg-type]
    with TestClient(app) as client:
        statuses = {
            client.post("/v1/chat", json={"message": "Hallo"}).status_code for _ in range(5)
        }

    assert statuses == {200}
    assert usage.days == {}


def test_exempt_accounts_are_neither_counted_nor_limited_by_the_day() -> None:
    usage = FakeUsage()
    settings = Settings(
        app_env="test",
        rate_limit_per_minute=0,
        daily_message_limit=1,
        daily_limit_exempt_users="owner",  # type: ignore[arg-type]  # as from the environment
    )
    app = create_app(settings, chat=FakeChat(), verifier=FakeVerifier(), usage=usage)  # type: ignore[arg-type]
    with TestClient(app) as client:
        owner = [
            client.post(
                "/v1/chat", json={"message": "Hallo"}, headers={"authorization": "Bearer owner"}
            ).status_code
            for _ in range(3)
        ]
        anna = [
            client.post(
                "/v1/chat", json={"message": "Hallo"}, headers={"authorization": "Bearer anna"}
            ).status_code
            for _ in range(2)
        ]

    assert owner == [200, 200, 200]
    assert anna == [200, 429]  # the same customer's data, but not the owner's account
    assert list(usage.days.values()) == [2]
