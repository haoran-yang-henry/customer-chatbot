"""Recorded model responses for deterministic, offline evaluation runs (CI gate).

`make eval-smoke-record` runs the smoke cases against the real providers and stores every chat
completion and embedding in a cassette (``eval/cassettes/smoke.json.gz``). CI replays it: the
agent code, the database, retrieval and scoring run for real; only the model answers come from
the file, so no API key and no LLM call is needed.

Keys depend on the request content only (messages, max tokens, temperature; the text of an
embedding), not on provider settings, so CI needs no knowledge of the recording machine's
configuration. A request that is not in the cassette raises ``ReplayMissError``: a changed prompt,
chunking or retrieval needs a new recording and must never fall back to a degraded answer.
"""

import gzip
import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from customer_agent.application.ports import ChatMessage, EmbeddingProvider, LLMProvider, LLMResult
from customer_agent.observability import get_logger, record_tokens

log = get_logger("cassette")

_RERECORD = "the request changed (prompt, evidence or chunking); run `make eval-smoke-record`"


class ReplayMissError(RuntimeError):
    """Not a ``ProviderUnavailableError`` on purpose: the agent must not degrade gracefully."""


def chat_key(messages: Sequence[ChatMessage], max_tokens: int, temperature: float) -> str:
    payload = [[m.model_dump() for m in messages], max_tokens, float(temperature)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def embedding_key(text: str, dimensions: int) -> str:
    return hashlib.sha256(f"{dimensions}|{text}".encode()).hexdigest()


class Cassette:
    def __init__(self, path: Path, *, record: bool) -> None:
        self.path = path
        self.record = record
        self._chat: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._dirty = False
        if path.exists():
            self._load()
        elif not record:
            raise FileNotFoundError(f"cassette {path} not found; {_RERECORD}")

    def _load(self) -> None:
        with gzip.open(self.path, "rt", encoding="utf-8") as file:
            data = json.load(file)
        self._chat.update(data["chat"])
        self._embeddings.update(data["embeddings"])

    def chat(self, key: str) -> LLMResult | None:
        value = self._chat.get(key)
        return LLMResult.model_validate(value) if value is not None else None

    def put_chat(self, key: str, result: LLMResult) -> None:
        self._chat[key] = result.model_dump()
        self._dirty = True

    def embedding(self, key: str) -> list[float] | None:
        return self._embeddings.get(key)

    def put_embedding(self, key: str, vector: list[float]) -> None:
        self._embeddings[key] = vector
        self._dirty = True

    def save(self) -> None:
        """Merge with what other processes recorded (ingest, then eval) and write
        reproducibly: sorted keys, gzip without timestamp or file name, so unchanged content keeps
        the same bytes."""
        if not (self.record and self._dirty):
            return
        if self.path.exists():
            current = (dict(self._chat), dict(self._embeddings))
            self._load()
            self._chat.update(current[0])
            self._embeddings.update(current[1])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            {"chat": self._chat, "embeddings": self._embeddings}, ensure_ascii=False, sort_keys=True
        )
        with (
            open(self.path, "wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as file,
        ):
            file.write(body.encode())
        self._dirty = False
        log.info(
            "cassette saved path=%s chat=%d embeddings=%d",
            self.path,
            len(self._chat),
            len(self._embeddings),
        )


@cache
def open_cassette(path: Path, record: bool) -> Cassette:
    """One instance per process, shared by the LLM roles and the embeddings."""
    return Cassette(path, record=record)


class CassetteLLM:
    """``LLMProvider``: answers from the cassette; in record mode, asks ``inner`` for misses."""

    def __init__(self, cassette: Cassette, inner: LLMProvider | None, model: str) -> None:
        self._cassette = cassette
        self._inner = inner
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def aclose(self) -> None:
        self._cassette.save()
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> LLMResult:
        key = chat_key(messages, max_tokens, temperature)
        recorded = self._cassette.chat(key)
        if recorded is not None:
            record_tokens(self._model, recorded.prompt_tokens, recorded.completion_tokens)
            return recorded
        if not self._cassette.record or self._inner is None:
            raise ReplayMissError(f"no recorded chat response {key[:12]}: {_RERECORD}")
        result = await self._inner.complete(
            messages, max_tokens=max_tokens, temperature=temperature, session_id=session_id
        )
        if result.text.strip():  # an empty generation is a failure, not a recording
            self._cassette.put_chat(key, result)
        return result

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        result = await self.complete(
            messages, max_tokens=max_tokens, temperature=temperature, session_id=session_id
        )
        yield result.text


class CassetteEmbeddings:
    """``EmbeddingProvider``: one recording per text, independent of how texts are batched."""

    def __init__(
        self, cassette: Cassette, inner: EmbeddingProvider | None, model: str, dimensions: int
    ) -> None:
        self._cassette = cassette
        self._inner = inner
        self._model = model
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def aclose(self) -> None:
        self._cassette.save()
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [embedding_key(text, self._dimensions) for text in texts]
        missing = [i for i, key in enumerate(keys) if self._cassette.embedding(key) is None]
        if missing:
            if not self._cassette.record or self._inner is None:
                raise ReplayMissError(f"{len(missing)} embedding(s) not recorded: {_RERECORD}")
            vectors = await self._inner.embed_documents([texts[i] for i in missing])
            for i, vector in zip(missing, vectors, strict=True):
                self._cassette.put_embedding(keys[i], vector)
        recorded = [self._cassette.embedding(key) for key in keys]
        return [v for v in recorded if v is not None]  # all present after recording misses

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
