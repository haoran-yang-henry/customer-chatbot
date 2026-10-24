"""Embedding provider adapters."""

from typing import Any, Self

import openai
from langchain_openai import OpenAIEmbeddings
from openai.resources.embeddings import AsyncEmbeddings
from openai.types import CreateEmbeddingResponse
from pydantic import model_validator

from customer_agent.application.ports import ProviderUnavailableError
from customer_agent.observability import step


class _OrderedAsyncEmbeddings(AsyncEmbeddings):
    """LangChain consumes response order; the provider's index defines input order."""

    async def create(self, **kwargs: Any) -> CreateEmbeddingResponse:
        response = await super().create(**kwargs)
        response.data.sort(key=lambda item: item.index)
        return response


class ProviderEmbeddings(OpenAIEmbeddings):
    """LangChain embeddings with the application's timeout and failure contract.

    Chunking already happens during ingestion: send the original strings, without LangChain's
    automatic tokenisation, splitting or vector averaging, so the existing index stays valid.
    """

    check_embedding_ctx_length: bool = False
    chunk_size: int = 256
    query_timeout: float | None = None

    @model_validator(mode="after")
    def _order_response(self) -> Self:
        if isinstance(self.async_client, AsyncEmbeddings):
            self.async_client = _OrderedAsyncEmbeddings(self.async_client._client)
        return self

    async def aclose(self) -> None:
        # LangChain keeps the SDK resource objects, not their owning root clients.
        try:
            await self.async_client._client.close()
        finally:
            self.client._client.close()

    async def aembed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs: Any
    ) -> list[list[float]]:
        try:
            with step("embed", model=self.model, texts=len(texts)):
                return await super().aembed_documents(texts, chunk_size, **kwargs)
        except openai.OpenAIError as exc:
            raise ProviderUnavailableError(f"{self.model}: {exc}") from exc

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        if self.query_timeout is not None:
            kwargs.setdefault("timeout", self.query_timeout)
        return await super().aembed_query(text, **kwargs)
