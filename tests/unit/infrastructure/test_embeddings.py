import asyncio
import json
import math
from pathlib import Path

import httpx2
import pytest
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from customer_agent.application.ports import ProviderUnavailableError
from customer_agent.infrastructure.cassette import Cassette, CassetteEmbeddings
from customer_agent.infrastructure.embeddings import HashingEmbeddings, ProviderEmbeddings
from customer_agent.infrastructure.resilience import BreakerEmbeddings, CircuitBreaker, CircuitState


async def test_hashing_embeddings_are_deterministic_and_normalised() -> None:
    provider = HashingEmbeddings(dimensions=64)

    first, second = await provider.aembed_documents(["Rücksendung", "Rücksendung"])

    assert first == second
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)


async def test_similar_texts_are_closer_than_unrelated_texts() -> None:
    provider = HashingEmbeddings(dimensions=256)
    query = await provider.aembed_query("Versandkosten")
    similar, unrelated = await provider.aembed_documents(["Versandkosten Österreich", "Akku"])

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(query, similar) > cosine(query, unrelated)


def embeddings(http: httpx2.AsyncClient) -> ProviderEmbeddings:
    return ProviderEmbeddings(
        openai_api_key=SecretStr("test"),
        openai_api_base="https://embeddings.test/v1",
        model="text-embedding-3-small",
        dimensions=2,
        chunk_size=2,
        request_timeout=15.0,
        query_timeout=5.0,
        max_retries=0,
        http_async_client=http,
    )


def response(request: httpx2.Request) -> httpx2.Response:
    texts = json.loads(request.content)["input"]
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [
                {"object": "embedding", "index": i, "embedding": [float(len(text)), 1.0]}
                for i, text in reversed(list(enumerate(texts)))
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        },
    )


async def test_langchain_sends_original_texts_in_batches_without_changing_vectors() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return response(request)

    model = embeddings(httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    assert isinstance(model, Embeddings)
    texts = ["Rücksendung\nÖsterreich", "<|endoftext|> {€}", "Versand"]
    try:
        vectors = await model.aembed_documents(texts)
        assert await model.aembed_documents([]) == []
    finally:
        await model.aclose()

    assert vectors == [[float(len(text)), 1.0] for text in texts]
    assert [json.loads(r.content)["input"] for r in requests] == [texts[:2], texts[2:]]
    for request in requests:
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-3-small" and body["dimensions"] == 2
        assert request.extensions["timeout"]["read"] == 15.0


async def test_query_timeout_survives_breaker_and_recording(tmp_path: Path) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return response(request)

    model = embeddings(httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    provider = CassetteEmbeddings(
        Cassette(tmp_path / "embeddings.json.gz", record=True),
        BreakerEmbeddings(model, CircuitBreaker("query-timeout")),
        model.model,
        2,
    )
    try:
        await provider.aembed_documents(["long batch"])
        first = await provider.aembed_query("Rückgabe")
        assert await provider.aembed_query("Rückgabe") == first
    finally:
        await provider.aclose()

    assert [r.extensions["timeout"]["read"] for r in requests] == [15.0, 5.0]


@pytest.mark.parametrize("operation", ["documents", "query"])
async def test_provider_errors_open_the_embedding_breaker(operation: str) -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(503, json={"error": {"message": "unavailable"}})

    model = embeddings(httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    breaker = CircuitBreaker("embedding-outage", failure_threshold=1)
    provider = BreakerEmbeddings(model, breaker)
    try:
        for _ in range(2):
            with pytest.raises(ProviderUnavailableError):
                if operation == "query":
                    await provider.aembed_query("Versand")
                else:
                    await provider.aembed_documents(["Versand"])
    finally:
        await provider.aclose()

    assert calls == 1 and breaker.state is CircuitState.OPEN


async def test_cancelled_query_releases_the_half_open_trial() -> None:
    class Cancelled(HashingEmbeddings):
        async def aembed_query(self, text: str) -> list[float]:
            raise asyncio.CancelledError

    breaker = CircuitBreaker("cancelled-query", failure_threshold=1, reset_after=0)
    breaker.record_failure()
    provider = BreakerEmbeddings(Cancelled(2), breaker)

    with pytest.raises(asyncio.CancelledError):
        await provider.aembed_query("Versand")

    assert breaker.state is CircuitState.HALF_OPEN
    breaker.before_call()
