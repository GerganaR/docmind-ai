"""Application settings, read from environment variables (and an optional .env file).

Python notes:
- ``@dataclass`` auto-generates ``__init__``, ``__repr__`` and equality for a class
  from its type-annotated fields - roughly a typed object literal / interface + constructor.
- ``frozen=True`` makes instances immutable (like ``Readonly<T>`` enforced at runtime).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    embedding_model: str
    chat_model: str
    docs_dir: Path
    chroma_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_distance: float

    def require_api_key(self) -> str:
        """Return the API key or fail with a helpful message (the key is never printed)."""
        if not self.openai_api_key:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or set it as an environment variable."
            )
        return self.openai_api_key


def load_settings() -> Settings:
    """Build Settings from the environment. Values in a local .env file are loaded first."""
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        embedding_model=os.getenv("DOCMIND_EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("DOCMIND_CHAT_MODEL", "gpt-4o-mini"),
        docs_dir=Path(os.getenv("DOCMIND_DOCS_DIR", PROJECT_ROOT / "data" / "sample_docs")),
        chroma_dir=Path(os.getenv("DOCMIND_CHROMA_DIR", PROJECT_ROOT / ".chroma")),
        collection_name=os.getenv("DOCMIND_COLLECTION", "docmind"),
        chunk_size=_int_env("DOCMIND_CHUNK_SIZE", 800),
        chunk_overlap=_int_env("DOCMIND_CHUNK_OVERLAP", 150),
        top_k=_int_env("DOCMIND_TOP_K", 4),
        max_distance=_float_env("DOCMIND_MAX_DISTANCE", 0.70),
    )
