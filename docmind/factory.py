"""Wire real dependencies (OpenAI + ChromaDB) together from Settings.

Keeping construction here means the pipeline itself never reads env vars or
creates network clients, which is what makes it easy to test with fakes.
"""

from __future__ import annotations

from openai import OpenAI

from docmind.config import Settings
from docmind.llm import OpenAIChatModel, OpenAIEmbedder
from docmind.pipeline import RAGPipeline
from docmind.vector_store import VectorStore


def make_openai_client(settings: Settings) -> OpenAI:
    # max_retries: the SDK retries rate limits / transient 5xx errors with backoff.
    return OpenAI(api_key=settings.require_api_key(), max_retries=3, timeout=60)


def make_store(settings: Settings) -> VectorStore:
    return VectorStore(settings.chroma_dir, settings.collection_name)


def make_embedder(settings: Settings, client: OpenAI) -> OpenAIEmbedder:
    return OpenAIEmbedder(client, settings.embedding_model)


def make_pipeline(settings: Settings, top_k: int | None = None) -> RAGPipeline:
    client = make_openai_client(settings)
    return RAGPipeline(
        embedder=make_embedder(settings, client),
        store=make_store(settings),
        llm=OpenAIChatModel(client, settings.chat_model),
        top_k=top_k or settings.top_k,
        max_distance=settings.max_distance,
    )
