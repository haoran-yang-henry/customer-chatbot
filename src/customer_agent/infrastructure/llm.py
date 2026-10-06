"""OpenAI-compatible chat completions adapter (OpenAI, OpenCode Go, Azure OpenAI, …), ADR-0008."""

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI

from customer_agent.application.ports import ChatMessage, LLMResult, ProviderUnavailableError
from customer_agent.observability import elapsed_ms, get_logger

log = get_logger("llm")


class OpenAICompatibleLLM:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        session_header: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._session_header = session_header
        self._reasoning_effort = reasoning_effort

    @property
    def model_name(self) -> str:
        return self._model

    def _request(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        temperature: float,
        session_id: str | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self._reasoning_effort:
            # Reasoning models reject max_tokens/temperature and need the newer parameters.
            request.pop("temperature")
            request["max_completion_tokens"] = request.pop("max_tokens")
            request["reasoning_effort"] = self._reasoning_effort
        if self._session_header and session_id:
            request["extra_headers"] = {self._session_header: session_id}
        return request

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> LLMResult:
        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                **self._request(messages, max_tokens, temperature, session_id)
            )
        except openai.OpenAIError as exc:
            log.warning(
                "model=%s error=%s took=%dms", self._model, type(exc).__name__, elapsed_ms(start)
            )
            raise ProviderUnavailableError(f"{self._model}: {exc}") from exc
        usage = response.usage
        result = LLMResult(
            text=response.choices[0].message.content or "",
            model=response.model or self._model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
        log.info(
            "model=%s tokens_in=%s tokens_out=%s took=%dms",
            result.model,
            result.prompt_tokens,
            result.completion_tokens,
            elapsed_ms(start),
        )
        return result

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        start = time.perf_counter()
        try:
            chunks = await self._client.chat.completions.create(
                **self._request(messages, max_tokens, temperature, session_id), stream=True
            )
            async for chunk in chunks:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.OpenAIError as exc:
            log.warning("model=%s stream error=%s", self._model, type(exc).__name__)
            raise ProviderUnavailableError(f"{self._model}: {exc}") from exc
        log.info("model=%s stream took=%dms", self._model, elapsed_ms(start))
