import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from prometheus_client import REGISTRY
from pydantic import SecretStr

from customer_agent.infrastructure.embeddings import ProviderEmbeddings
from customer_agent.infrastructure.lifecycle import aclose_all
from customer_agent.infrastructure.models import (
    BreakerChatModel,
    CachedChatModel,
    ProviderChatModel,
)
from customer_agent.infrastructure.prompts import load_prompt
from customer_agent.infrastructure.resilience import CircuitBreaker
from tests.fakes import FakeLLM

MESSAGES: list[BaseMessage] = [HumanMessage("Hallo")]


async def ask(
    model: BaseChatModel, messages: list[BaseMessage] = MESSAGES, session_id: str | None = None
) -> BaseMessage:
    return await model.ainvoke(messages, max_tokens=10, temperature=0.0, session_id=session_id)


def provider(reasoning_effort: str | None = None) -> ProviderChatModel:
    return ProviderChatModel(
        model_name="model-x", openai_api_key=SecretStr("test"), reasoning_effort=reasoning_effort
    )


def test_standard_request_parameters() -> None:
    payload = provider()._get_request_payload(MESSAGES, max_tokens=100, temperature=0.1)

    assert payload["max_completion_tokens"] == 100 and payload["temperature"] == 0.1
    assert "extra_headers" not in payload


def test_reasoning_models_get_no_temperature() -> None:
    payload = provider("none")._get_request_payload(MESSAGES, max_tokens=100, temperature=0.1)

    assert "temperature" not in payload
    assert payload["max_completion_tokens"] == 100 and payload["reasoning_effort"] == "none"


async def test_cache_answers_identical_requests_from_disk(tmp_path: Path) -> None:
    inner = FakeLLM("Antwort")
    cache = CachedChatModel(inner=inner, directory=tmp_path, namespace="n")

    first = await ask(cache)
    second = await ask(CachedChatModel(inner=FakeLLM("anders"), directory=tmp_path, namespace="n"))
    await ask(cache, [HumanMessage("Neu")])

    assert first.text == second.text == "Antwort"
    assert len(inner.calls) == 2  # third request differs → real call


def test_cache_keys_are_those_of_the_day_3_cache(tmp_path: Path) -> None:
    """Existing .cache/llm entries stay valid across the rebuild."""
    cache = CachedChatModel(inner=FakeLLM(), directory=tmp_path, namespace="n")
    key = json.dumps(
        ["n", [{"role": "user", "content": "Hallo"}], 10, 0.0], ensure_ascii=False
    ).encode()

    path = cache._path(MESSAGES, {"max_tokens": 10, "temperature": 0.0})

    assert path.name == f"{hashlib.sha256(key).hexdigest()}.json"


async def test_empty_generation_is_not_cached(tmp_path: Path) -> None:
    await ask(CachedChatModel(inner=FakeLLM(""), directory=tmp_path, namespace="n"))

    assert list(tmp_path.iterdir()) == []


async def test_cache_forwards_the_session_on_a_miss(tmp_path: Path) -> None:
    inner = FakeLLM("Antwort")

    await ask(CachedChatModel(inner=inner, directory=tmp_path, namespace="n"), session_id="conv-1")

    assert inner.session_ids == ["conv-1"]


async def test_cache_is_separated_by_namespace(tmp_path: Path) -> None:
    """Changing e.g. reasoning settings must not reuse answers produced with the old ones."""
    await ask(CachedChatModel(inner=FakeLLM("alt"), directory=tmp_path, namespace="a"))
    inner = FakeLLM("neu")

    result = await ask(CachedChatModel(inner=inner, directory=tmp_path, namespace="b"))

    assert result.text == "neu" and len(inner.calls) == 1


@pytest.mark.parametrize(
    ("name", "version"),
    [
        ("answer", 1),
        ("answer", 2),
        ("answer", 3),
        ("router", 1),
        ("router", 2),
        ("router", 3),
        ("router", 4),
        ("judge", 1),
        ("guard", 1),
    ],
)
def test_shipped_prompts_load(name: str, version: int) -> None:
    prompt = load_prompt(name, version)

    assert prompt.ref == f"{name}/v{version}"


def test_prompt_rendering_requires_all_placeholders() -> None:
    prompt = load_prompt("answer", 1)

    with pytest.raises(KeyError):
        prompt.render_messages(evidence="…")


async def test_models_close_their_http_clients() -> None:
    llm_http = httpx2.AsyncClient()
    model = ProviderChatModel(
        model_name="m", openai_api_key=SecretStr("t"), http_async_client=llm_http
    )
    embed_http = httpx2.AsyncClient()
    embeddings = ProviderEmbeddings(
        openai_api_key=SecretStr("t"), model="e", dimensions=8, http_async_client=embed_http
    )

    await aclose_all(model, embeddings, None, "not closable")

    assert llm_http.is_closed and embed_http.is_closed
    assert model.root_client is not None and model.root_client.is_closed()
    assert embeddings.client._client.is_closed()


# --- the real SDK against a mocked transport ------------------------------------------------

Handler = Callable[[httpx2.Request], httpx2.Response]


def mocked(
    handler: Handler, model: str = "model-x", session_header: str | None = None
) -> ProviderChatModel:
    return ProviderChatModel(
        model_name=model,
        openai_api_key=SecretStr("test"),
        openai_api_base="https://llm.test/v1",
        http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
        session_header=session_header,
        stream_usage=True,
    )


def completion(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "model-x",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hallo!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        },
    )


def stream(*, usage_twice: bool = False) -> Handler:
    """OpenAI reports usage on a final chunk without choices; OpenCode Go also on the finishing
    chunk (rebuild plan §9, phase 0)."""

    def chunk(
        delta: dict[str, str], usage: dict[str, int] | None = None, finish: str | None = None
    ) -> str:
        body = {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "model-y",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
            if delta or finish
            else [],
            "usage": usage,
        }
        return f"data: {json.dumps(body)}\n\n"

    usage = {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22}
    events = [
        chunk({"role": "assistant", "content": "Hal"}),
        chunk({"content": "lo"}),
        chunk({}, usage if usage_twice else None, finish="stop"),
        chunk({}, usage),
        "data: [DONE]\n\n",
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content="".join(events).encode()
        )

    return handler


def sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


async def test_the_session_header_goes_with_every_call_through_the_wrappers() -> None:
    """OpenCode Go answers 400 without it (rebuild plan §9, phase 0)."""
    headers: list[str | None] = []

    def recording(request: httpx2.Request) -> httpx2.Response:
        headers.append(request.headers.get("x-opencode-session"))
        streaming = json.loads(request.content).get("stream")
        return stream()(request) if streaming else completion(request)

    model = BreakerChatModel(
        inner=mocked(recording, session_header="x-opencode-session"),
        breaker=CircuitBreaker("session-header-test"),
    )

    await ask(model, session_id="s1")
    await ask(model)  # a stateless caller sends none
    async for _ in model.astream(MESSAGES, max_tokens=10, session_id="s2"):
        pass  # LangChain's astream passes no run manager: the id must travel as a keyword

    assert headers == ["s1", None, "s2"]


async def test_completion_records_tokens() -> None:
    before = sample("ca_llm_tokens_total", model="model-x", kind="prompt")

    result = await ask(mocked(completion))

    assert result.text == "Hallo!"
    assert sample("ca_llm_tokens_total", model="model-x", kind="prompt") == before + 12


@pytest.mark.parametrize("usage_twice", [False, True])
async def test_stream_records_time_to_first_token_and_counts_usage_once(
    usage_twice: bool,
) -> None:
    ttft = sample("ca_llm_time_to_first_token_seconds_count", model="model-y")
    tokens = sample("ca_llm_tokens_total", model="model-y", kind="completion")

    final = None
    async for chunk in mocked(stream(usage_twice=usage_twice), "model-y").astream(
        MESSAGES, max_tokens=10
    ):
        final = chunk if final is None else final + chunk

    assert final is not None and final.text == "Hallo"
    assert final.usage_metadata is not None
    assert final.usage_metadata["output_tokens"] == 2  # not 4 when reported twice
    assert sample("ca_llm_time_to_first_token_seconds_count", model="model-y") == ttft + 1
    assert sample("ca_llm_tokens_total", model="model-y", kind="completion") == tokens + 2


async def test_cache_hits_count_the_recorded_usage(tmp_path: Path) -> None:
    """Evaluation runs mostly hit the cache; their token totals still estimate the cost."""
    inner = FakeLLM("Antwort", usage=(100, 10))
    cache = CachedChatModel(inner=inner, directory=tmp_path, namespace="n")
    await ask(cache)  # miss: stored with usage
    before = sample("ca_llm_tokens_total", model="fake-llm", kind="prompt")

    hit = await ask(cache)

    assert hit.text == "Antwort" and len(inner.calls) == 1
    assert sample("ca_llm_tokens_total", model="fake-llm", kind="prompt") == before + 100
