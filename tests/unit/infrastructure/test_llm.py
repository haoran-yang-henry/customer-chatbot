import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from langchain_core.messages import HumanMessage
from prometheus_client import REGISTRY
from pydantic import SecretStr

from customer_agent.application.ports import ChatMessage, LLMResult
from customer_agent.infrastructure.embeddings import ProviderEmbeddings
from customer_agent.infrastructure.lifecycle import aclose_all
from customer_agent.infrastructure.models import (
    CachedChatModel,
    ChatModelLLM,
    ProviderChatModel,
)
from customer_agent.infrastructure.prompts import load_prompt
from tests.fakes import FakeChatModel

MESSAGES = [ChatMessage(role="user", content="Hallo")]


def provider(reasoning_effort: str | None = None) -> ProviderChatModel:
    return ProviderChatModel(
        model_name="model-x", openai_api_key=SecretStr("test"), reasoning_effort=reasoning_effort
    )


def test_standard_request_parameters() -> None:
    payload = provider()._get_request_payload(
        [HumanMessage("Hallo")], max_tokens=100, temperature=0.1
    )

    assert payload["max_completion_tokens"] == 100 and payload["temperature"] == 0.1
    assert "extra_headers" not in payload


def test_reasoning_models_get_no_temperature() -> None:
    payload = provider("none")._get_request_payload(
        [HumanMessage("Hallo")], max_tokens=100, temperature=0.1
    )

    assert "temperature" not in payload
    assert payload["max_completion_tokens"] == 100 and payload["reasoning_effort"] == "none"


async def test_the_session_header_goes_with_every_call() -> None:
    model = FakeChatModel(reply="Hallo!", usage=(12, 3))
    llm = ChatModelLLM(model, session_header="x-opencode-session")

    result = await llm.complete(MESSAGES, max_tokens=10, temperature=0.0, session_id="s1")
    await llm.complete(MESSAGES, max_tokens=10, temperature=0.0)  # stateless caller

    assert (result.text, result.prompt_tokens, result.completion_tokens) == ("Hallo!", 12, 3)
    assert [kwargs for _, kwargs in model.calls] == [
        {"max_tokens": 10, "temperature": 0.0, "extra_headers": {"x-opencode-session": "s1"}},
        {"max_tokens": 10, "temperature": 0.0},
    ]


async def test_cache_answers_identical_requests_from_disk(tmp_path: Path) -> None:
    inner = FakeChatModel(reply="Antwort")
    cache = ChatModelLLM(CachedChatModel(inner=inner, directory=tmp_path, namespace="n"))

    first = await cache.complete(MESSAGES, max_tokens=10, temperature=0.0)
    second = await ChatModelLLM(
        CachedChatModel(inner=FakeChatModel(reply="anders"), directory=tmp_path, namespace="n")
    ).complete(MESSAGES, max_tokens=10, temperature=0.0)
    await cache.complete([ChatMessage(role="user", content="Neu")], max_tokens=10, temperature=0.0)

    assert first.text == second.text == "Antwort"
    assert len(inner.calls) == 2  # third request differs → real call


def test_cache_keys_are_those_of_the_day_3_cache(tmp_path: Path) -> None:
    """Existing .cache/llm entries stay valid across the rebuild."""
    cache = CachedChatModel(inner=FakeChatModel(), directory=tmp_path, namespace="n")
    key = json.dumps(
        ["n", [{"role": "user", "content": "Hallo"}], 10, 0.0], ensure_ascii=False
    ).encode()

    path = cache._path([HumanMessage("Hallo")], {"max_tokens": 10, "temperature": 0.0})

    assert path.name == f"{hashlib.sha256(key).hexdigest()}.json"


async def test_empty_generation_is_not_cached(tmp_path: Path) -> None:
    await ChatModelLLM(
        CachedChatModel(inner=FakeChatModel(reply=""), directory=tmp_path, namespace="n")
    ).complete(MESSAGES, max_tokens=10, temperature=0.0)

    assert list(tmp_path.iterdir()) == []


async def test_cache_forwards_the_session_header_on_a_miss(tmp_path: Path) -> None:
    inner = FakeChatModel(reply="Antwort")

    await ChatModelLLM(
        CachedChatModel(inner=inner, directory=tmp_path, namespace="n"),
        session_header="x-opencode-session",
    ).complete(MESSAGES, max_tokens=10, temperature=0.0, session_id="conv-1")

    assert inner.calls[0][1]["extra_headers"] == {"x-opencode-session": "conv-1"}


async def test_cache_is_separated_by_namespace(tmp_path: Path) -> None:
    """Changing e.g. reasoning settings must not reuse answers produced with the old ones."""
    await ChatModelLLM(
        CachedChatModel(inner=FakeChatModel(reply="alt"), directory=tmp_path, namespace="a")
    ).complete(MESSAGES, max_tokens=10, temperature=0.0)
    inner = FakeChatModel(reply="neu")

    result = await ChatModelLLM(
        CachedChatModel(inner=inner, directory=tmp_path, namespace="b")
    ).complete(MESSAGES, max_tokens=10, temperature=0.0)

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
        prompt.render_user(evidence="…")


async def test_models_close_their_http_clients() -> None:
    llm_http = httpx2.AsyncClient()
    model = ProviderChatModel(
        model_name="m", openai_api_key=SecretStr("t"), http_async_client=llm_http
    )
    embed_http = httpx2.AsyncClient()
    embed_model = ProviderEmbeddings(
        openai_api_key=SecretStr("t"), model="e", dimensions=8, http_async_client=embed_http
    )
    embeddings = embed_model

    await aclose_all(ChatModelLLM(model), embeddings, None, "not closable")

    assert llm_http.is_closed and embed_http.is_closed
    assert model.root_client is not None and model.root_client.is_closed()
    assert embed_model.client._client.is_closed()


# --- the real SDK against a mocked transport ------------------------------------------------

Handler = Callable[[httpx2.Request], httpx2.Response]


def mocked(handler: Handler, model: str = "model-x") -> ChatModelLLM:
    return ChatModelLLM(
        ProviderChatModel(
            model_name=model,
            openai_api_key=SecretStr("test"),
            openai_api_base="https://llm.test/v1",
            http_async_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
            max_retries=0,
            stream_usage=True,
        )
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


async def test_completion_records_tokens() -> None:
    before = sample("ca_llm_tokens_total", model="model-x", kind="prompt")

    result = await mocked(completion).complete(MESSAGES, max_tokens=10, temperature=0.0)

    assert (result.text, result.prompt_tokens, result.completion_tokens) == ("Hallo!", 12, 3)
    assert sample("ca_llm_tokens_total", model="model-x", kind="prompt") == before + 12


@pytest.mark.parametrize("usage_twice", [False, True])
async def test_stream_records_time_to_first_token_and_counts_usage_once(
    usage_twice: bool,
) -> None:
    ttft = sample("ca_llm_time_to_first_token_seconds_count", model="model-y")
    tokens = sample("ca_llm_tokens_total", model="model-y", kind="completion")
    llm = mocked(stream(usage_twice=usage_twice), "model-y")

    text = "".join([d async for d in llm.stream(MESSAGES, max_tokens=10, temperature=0)])
    final = None
    async for chunk in llm.chat_model.astream([HumanMessage("Hallo")], max_tokens=10):
        final = chunk if final is None else final + chunk

    assert text == "Hallo"
    assert sample("ca_llm_time_to_first_token_seconds_count", model="model-y") == ttft + 2
    assert sample("ca_llm_tokens_total", model="model-y", kind="completion") == tokens + 4
    assert final is not None and final.usage_metadata is not None
    assert final.usage_metadata["output_tokens"] == 2  # not 4 when reported twice


async def test_cache_hits_count_the_recorded_usage(tmp_path: Path) -> None:
    """Evaluation runs mostly hit the cache; their token totals still estimate the cost."""
    inner = FakeChatModel(reply="Antwort", usage=(100, 10))
    cache = ChatModelLLM(CachedChatModel(inner=inner, directory=tmp_path, namespace="n"))
    await cache.complete(MESSAGES, max_tokens=10, temperature=0.0)  # miss: stored with usage
    before = sample("ca_llm_tokens_total", model="fake-chat", kind="prompt")

    hit = await cache.complete(MESSAGES, max_tokens=10, temperature=0.0)

    assert isinstance(hit, LLMResult) and hit.prompt_tokens == 100
    assert sample("ca_llm_tokens_total", model="fake-chat", kind="prompt") == before + 100
