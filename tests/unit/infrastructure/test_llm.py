from pathlib import Path

import pytest
from openai import AsyncOpenAI

from customer_agent.application.ports import ChatMessage
from customer_agent.infrastructure.llm import OpenAICompatibleLLM
from customer_agent.infrastructure.llm_cache import CachingLLM
from customer_agent.infrastructure.prompts import load_prompt
from tests.fakes import FakeLLM

MESSAGES = [ChatMessage(role="user", content="Hallo")]


def adapter(**kwargs: str) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(AsyncOpenAI(api_key="test"), "model-x", **kwargs)


def test_standard_request_parameters() -> None:
    request = adapter()._request(MESSAGES, 100, 0.1, session_id="s1")

    assert request["max_tokens"] == 100 and request["temperature"] == 0.1
    assert "extra_headers" not in request  # no session header configured


def test_session_header_is_sent_when_configured() -> None:
    request = adapter(session_header="x-opencode-session")._request(MESSAGES, 100, 0.1, "s1")

    assert request["extra_headers"] == {"x-opencode-session": "s1"}


def test_reasoning_models_use_newer_parameters() -> None:
    request = adapter(reasoning_effort="none")._request(MESSAGES, 100, 0.1, None)

    assert "temperature" not in request and "max_tokens" not in request
    assert request["max_completion_tokens"] == 100 and request["reasoning_effort"] == "none"


async def test_cache_answers_identical_requests_from_disk(tmp_path: Path) -> None:
    inner = FakeLLM("Antwort")
    cache = CachingLLM(inner, tmp_path)

    first = await cache.complete(MESSAGES, max_tokens=10, temperature=0.0)
    second = await CachingLLM(FakeLLM("anders"), tmp_path).complete(
        MESSAGES, max_tokens=10, temperature=0.0
    )
    await cache.complete([ChatMessage(role="user", content="Neu")], max_tokens=10, temperature=0.0)

    assert first.text == second.text == "Antwort"
    assert len(inner.calls) == 2  # third request differs → real call


@pytest.mark.parametrize(
    ("name", "version"), [("answer", 1), ("answer", 2), ("router", 1), ("router", 2), ("judge", 1)]
)
def test_shipped_prompts_load(name: str, version: int) -> None:
    prompt = load_prompt(name, version)

    assert prompt.ref == f"{name}/v{version}"


def test_prompt_rendering_requires_all_placeholders() -> None:
    prompt = load_prompt("answer", 1)

    with pytest.raises(KeyError):
        prompt.render_user(evidence="…")


async def test_empty_generation_is_not_cached(tmp_path: Path) -> None:
    await CachingLLM(FakeLLM(""), tmp_path).complete(MESSAGES, max_tokens=10, temperature=0.0)

    assert list(tmp_path.iterdir()) == []


async def test_cache_forwards_session_id_on_a_miss(tmp_path: Path) -> None:
    inner = FakeLLM("Antwort")

    await CachingLLM(inner, tmp_path).complete(
        MESSAGES, max_tokens=10, temperature=0.0, session_id="conv-1"
    )

    assert inner.session_ids == ["conv-1"]
