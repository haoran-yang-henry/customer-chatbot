"""Embedding provider adapters."""

import hashlib
import math
from collections.abc import Sequence

import openai
from openai import AsyncOpenAI

from customer_agent.application.ports import ProviderUnavailableError


class OpenAIEmbeddingProvider:
    """OpenAI embeddings API (ADR-0007). The SDK retries transient errors itself."""

    def __init__(
        self, client: AsyncOpenAI, model: str, dimensions: int, batch_size: int = 256
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def aclose(self) -> None:
        await self._client.close()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                response = await self._client.embeddings.create(
                    model=self._model, input=batch, dimensions=self._dimensions
                )
            except openai.OpenAIError as exc:
                raise ProviderUnavailableError(f"{self._model}: {exc}") from exc
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class HashingEmbeddingProvider:
    """Deterministic, offline stand-in for tests: hashed character trigrams, L2-normalised.

    Captures lexical overlap only — good enough to exercise the pipeline, useless for measuring
    retrieval quality. Uses blake2b because Python's built-in ``hash`` is salted per process.
    """

    def __init__(self, dimensions: int, model_name: str = "hashing-trigram-v1") -> None:
        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        padded = f"  {text.casefold()}  "
        for i in range(len(padded) - 2):
            digest = hashlib.blake2b(padded[i : i + 3].encode(), digest_size=8).digest()
            vector[int.from_bytes(digest) % self._dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
