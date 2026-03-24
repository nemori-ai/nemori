"""Async embedding client implementing EmbeddingProvider."""
from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from nemori.domain.exceptions import EmbeddingError

logger = logging.getLogger("nemori")


class AsyncEmbeddingClient:
    """Async embedding generation via OpenAI-compatible API."""

    def __init__(
        self, api_key: str, model: str = "text-embedding-3-small",
        base_url: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dimensions = dimensions

    def _extra_kwargs(self) -> dict[str, Any]:
        """Build optional kwargs for the embeddings API call."""
        if self._dimensions is not None:
            return {"dimensions": self._dimensions}
        return {}

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=text, **self._extra_kwargs()
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

    async def probe_dimension(self) -> int:
        """Probe actual embedding dimension by sending a test string."""
        vec = await self.embed("dimension probe")
        return len(vec)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                model=self._model, input=texts, **self._extra_kwargs()
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(f"Batch embedding failed: {e}") from e
