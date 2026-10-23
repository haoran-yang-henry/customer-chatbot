"""Record and replay of model responses for the offline CI gate (Day 8)."""

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from customer_agent.application.ports import ChatMessage, ProviderUnavailableError
from customer_agent.infrastructure.cassette import (
    Cassette,
    CassetteEmbeddings,
    ReplayMissError,
    chat_key,
)
from customer_agent.infrastructure.models import ChatModelLLM, ReplayChatModel
from tests.fakes import FakeChatModel

MESSAGES = [ChatMessage(role="system", content="Regeln"), ChatMessage(role="user", content="Hallo")]


class CountingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]


def replay_llm(cassette: Cassette, inner: FakeChatModel | None = None) -> ChatModelLLM:
    return ChatModelLLM(ReplayChatModel(cassette=cassette, inner=inner, label="answer-model"))


async def record_chat(path: Path, reply: str = "Guten Tag") -> FakeChatModel:
    inner = FakeChatModel(reply=reply)
    llm = replay_llm(Cassette(path, record=True), inner)
    await llm.complete(MESSAGES, max_tokens=100, temperature=0.0)
    await llm.aclose()
    return inner


async def test_replays_a_recorded_completion_without_a_provider(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json.gz"
    inner = await record_chat(path)

    replay = replay_llm(Cassette(path, record=False), None)
    result = await replay.complete(MESSAGES, max_tokens=100, temperature=0.0)
    streamed = [piece async for piece in replay.stream(MESSAGES, max_tokens=100, temperature=0.0)]

    assert result.text == "Guten Tag"
    assert streamed == ["Guten Tag"]
    assert len(inner.calls) == 1


async def test_a_changed_request_is_a_miss_not_a_degraded_answer(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json.gz"
    await record_chat(path)
    replay = replay_llm(Cassette(path, record=False), None)
    changed = [*MESSAGES[:1], ChatMessage(role="user", content="Hallo!")]

    with pytest.raises(ReplayMissError, match="eval-smoke-record") as error:
        await replay.complete(changed, max_tokens=100, temperature=0.0)
    # the orchestrator turns provider outages into a handoff; a miss must fail the run instead
    assert not isinstance(error.value, ProviderUnavailableError)


def test_keys_depend_on_every_request_parameter() -> None:
    base = chat_key(MESSAGES, 100, 0.0)

    assert chat_key(MESSAGES, 100, 0.0) == base
    assert chat_key(MESSAGES, 200, 0.0) != base
    assert chat_key(MESSAGES, 100, 0.2) != base
    assert chat_key(MESSAGES[1:], 100, 0.0) != base


def test_replay_requires_the_cassette(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="eval-smoke-record"):
        Cassette(tmp_path / "missing.json.gz", record=False)


async def test_an_empty_generation_is_not_recorded(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json.gz"
    await record_chat(path, reply="  ")

    assert not path.exists()


async def test_embeddings_are_recorded_per_text_whatever_the_batching(tmp_path: Path) -> None:
    path = tmp_path / "smoke.json.gz"
    inner = CountingEmbeddings()
    recorder = CassetteEmbeddings(Cassette(path, record=True), inner, "fake-embeddings", 2)
    await recorder.aembed_documents(["ein", "zwei"])
    await recorder.aembed_documents(["zwei", "dreizehn"])
    await recorder.aclose()

    replay = CassetteEmbeddings(Cassette(path, record=False), None, "fake-embeddings", 2)

    assert inner.batches == [["ein", "zwei"], ["dreizehn"]]  # only misses reach the provider
    assert await replay.aembed_documents(["dreizehn", "ein"]) == [[8.0, 1.0], [3.0, 1.0]]
    assert await replay.aembed_query("zwei") == [4.0, 1.0]
    with pytest.raises(ReplayMissError):
        await replay.aembed_query("vier")


async def test_recordings_of_separate_processes_are_merged(tmp_path: Path) -> None:
    # ingest records the chunk embeddings, the evaluation run then adds the chat responses
    path = tmp_path / "smoke.json.gz"
    ingest = CassetteEmbeddings(Cassette(path, record=True), CountingEmbeddings(), "fake", 2)
    await ingest.aembed_documents(["Rückgabe innerhalb von 30 Tagen"])
    await ingest.aclose()
    await record_chat(path)

    cassette = Cassette(path, record=False)
    replay = CassetteEmbeddings(cassette, None, "fake", 2)

    assert await replay.aembed_query("Rückgabe innerhalb von 30 Tagen")
    llm = replay_llm(cassette, None)
    assert await llm.complete(MESSAGES, max_tokens=100, temperature=0.0)


async def test_unchanged_recordings_keep_the_same_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    await record_chat(first)
    await record_chat(second)

    assert first.read_bytes() == second.read_bytes()


async def test_langchain_replay_stream_preserves_usage(tmp_path: Path) -> None:
    path = tmp_path / "usage.json.gz"
    recorder = replay_llm(
        Cassette(path, record=True), FakeChatModel(reply="Antwort", usage=(12, 3))
    )
    await recorder.complete(MESSAGES, max_tokens=100, temperature=0.0)
    await recorder.aclose()
    model = ReplayChatModel(cassette=Cassette(path, record=False), label="answer-model")

    chunks = [
        chunk
        async for chunk in model.astream(
            [SystemMessage("Regeln"), HumanMessage("Hallo")], max_tokens=100, temperature=0.0
        )
    ]

    assert "".join(chunk.text for chunk in chunks) == "Antwort"
    usages = [chunk.usage_metadata for chunk in chunks if chunk.usage_metadata]
    assert usages == [{"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}]
