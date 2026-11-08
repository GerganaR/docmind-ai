"""Shared pytest fixtures and fakes.

Python notes:
- pytest auto-discovers this file; fixtures defined here are available to every test.
- A *fixture* is a function whose return value is injected into any test that names it
  as a parameter (dependency injection by argument name). ``tmp_path`` is a built-in
  fixture giving each test its own empty temporary folder.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Sequence

import pytest

from docmind.vector_store import VectorStore

DIM = 128
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "and", "in", "for", "how", "what",
              "do", "does", "many", "per", "on", "be", "with", "can", "i", "my"}


class FakeEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors.

    Not semantic like OpenAI embeddings, but good enough to test retrieval wiring
    without network calls. Uses crc32 (not Python's hash(), which is randomized per run).
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * DIM
        vec[0] = 0.05  # small constant so no vector is all zeros
        for word in _WORD_RE.findall(text.lower()):
            if word not in _STOPWORDS:
                vec[1 + zlib.crc32(word.encode()) % (DIM - 1)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]


class FakeChatModel:
    """Returns a canned reply and records the prompts it received."""

    def __init__(self, reply: str = "Employees get 28 vacation days per year [S1].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", collection_name="test")


@pytest.fixture
def docs_dir(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "handbook.md").write_text(
        "# Handbook\n\nVacation: full-time employees receive 28 vacation days per calendar year.\n\n"
        "Remote work is allowed up to three days per week.",
        encoding="utf-8",
    )
    (folder / "security.txt").write_text(
        "Passwords must be at least 14 characters. Report phishing incidents within one hour.",
        encoding="utf-8",
    )
    return folder
