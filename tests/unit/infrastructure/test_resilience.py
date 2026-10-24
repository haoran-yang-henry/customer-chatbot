"""Circuit breaker and provider failure handling (Day 7, NFR-11)."""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import cast

import httpx2
import pytest
from langchain_core.messages import BaseMessageChunk, HumanMessage
from prometheus_client import REGISTRY
from pydantic import SecretStr

from customer_agent.application.ports import ProviderUnavailableError
from customer_agent.infrastructure.models import BreakerChatModel, ProviderChatModel
from customer_agent.infrastructure.resilience import CircuitBreaker, CircuitState
from tests.fakes import FakeLLM

MESSAGES = [HumanMessage("Hallo")]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def breaker(clock: Clock, dependency: str = "test-llm") -> CircuitBreaker:
    return CircuitBreaker(dependency, failure_threshold=3, reset_after=30.0, clock=clock)


def test_opens_after_consecutive_failures_and_fails_fast() -> None:
    cb = breaker(Clock())
    for _ in range(3):
        cb.before_call()
        cb.record_failure()

    assert cb.state is CircuitState.OPEN
    with pytest.raises(ProviderUnavailableError, match="circuit open"):
        cb.before_call()
    assert REGISTRY.get_sample_value("ca_circuit_open", {"dependency": "test-llm"}) == 1


def test_a_success_resets_the_failure_count() -> None:
    cb = breaker(Clock())
    for outcome in ("fail", "fail", "ok", "fail", "fail"):
        cb.before_call()
        cb.record_failure() if outcome == "fail" else cb.record_success()

    assert cb.state is CircuitState.CLOSED


def test_half_open_lets_exactly_one_trial_through() -> None:
    clock = Clock()
    cb = breaker(clock)
    for _ in range(3):
        cb.record_failure()
    clock.now += 30

    cb.before_call()  # the trial
    with pytest.raises(ProviderUnavailableError):
        cb.before_call()  # everyone else still fails fast
    cb.record_success()

    assert cb.state is CircuitState.CLOSED
    cb.before_call()


def test_a_failed_trial_reopens_for_another_period() -> None:
    clock = Clock()
    cb = breaker(clock)
    for _ in range(3):
        cb.record_failure()
    clock.now += 30
    cb.before_call()
    cb.record_failure()

    assert cb.state is CircuitState.OPEN
    clock.now += 29
    assert cb.state is CircuitState.OPEN


def test_an_abandoned_trial_does_not_block_the_breaker_forever() -> None:
    clock = Clock()
    cb = breaker(clock)
    for _ in range(3):
        cb.record_failure()
    clock.now += 30
    cb.before_call()
    cb.abandon()  # e.g. the client disconnected while the trial streamed

    cb.before_call()  # a new trial is allowed


async def test_open_breaker_skips_the_provider() -> None:
    inner = FakeLLM(fail=True)
    llm = BreakerChatModel(inner=inner, breaker=breaker(Clock(), "skip-llm"))
    for _ in range(3):
        with pytest.raises(ProviderUnavailableError):
            await llm.ainvoke(MESSAGES, max_tokens=5, temperature=0)

    with pytest.raises(ProviderUnavailableError, match="circuit open"):
        await llm.ainvoke(MESSAGES, max_tokens=5, temperature=0)
    assert len(inner.calls) == 3  # the fourth call never reached the provider


async def test_stream_closed_early_frees_a_half_open_trial() -> None:
    clock = Clock()
    cb = breaker(clock, "stream-llm")
    for _ in range(3):
        cb.record_failure()
    clock.now += 30
    llm = BreakerChatModel(inner=FakeLLM("Eine lange Antwort", piece=3), breaker=cb)

    stream = cast(
        AsyncGenerator[BaseMessageChunk], llm.astream(MESSAGES, max_tokens=5, temperature=0)
    )
    await anext(stream)
    await stream.aclose()
    # LangChain's astream does not close the stream it reads; the event loop's async-generator
    # finalizer does, on its next turn.
    await asyncio.sleep(0)

    assert cb.state is CircuitState.HALF_OPEN
    cb.before_call()


# --- the real SDK: timeouts and server errors become ProviderUnavailableError ----------------


def sdk_llm(handler: httpx2.MockTransport, timeout: float = 0.2) -> ProviderChatModel:
    return ProviderChatModel(
        model_name="model-z",
        openai_api_key=SecretStr("test"),
        openai_api_base="https://llm.test/v1",
        http_async_client=httpx2.AsyncClient(transport=handler),
        request_timeout=timeout,
        max_retries=1,
    )


async def test_server_errors_are_retried_once_then_reported() -> None:
    calls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request.url.path)
        return httpx2.Response(503, json={"error": {"message": "overloaded"}})

    with pytest.raises(ProviderUnavailableError):
        await sdk_llm(httpx2.MockTransport(handler)).ainvoke(MESSAGES, max_tokens=5, temperature=0)
    assert len(calls) == 2  # one retry with backoff, then the fallback takes over


async def test_a_hanging_provider_fails_within_the_timeout_budget() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("no response", request=request)

    start = time.perf_counter()
    with pytest.raises(ProviderUnavailableError):
        await sdk_llm(httpx2.MockTransport(handler)).ainvoke(MESSAGES, max_tokens=5, temperature=0)
    assert time.perf_counter() - start < 5.0  # retry backoff included
