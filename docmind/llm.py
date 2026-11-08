"""Thin wrappers around the OpenAI API (embeddings + chat completions).

Python note: ``typing.Protocol`` describes a shape, like a TypeScript ``interface``.
Any class with matching methods satisfies it (structural typing) - no ``implements``
needed. The tests use this to swap in fake models without touching the network.
"""

from __future__ import annotations

from typing import Protocol, Sequence

from openai import OpenAI


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenAIEmbedder:
    def __init__(self, client: OpenAI, model: str, batch_size: int = 100):
        self.client = client
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        # Send texts in batches to stay well within request size limits.
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            # Sort by index so output order always matches input order.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return vectors


class OpenAIChatModel:
    def __init__(self, client: OpenAI, model: str, temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,  # 0 = as deterministic as possible
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()
