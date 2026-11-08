"""ChromaDB persistence: store chunk embeddings + metadata and run similarity search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    start: int
    end: int


@dataclass(frozen=True)
class SearchResult:
    chunk: StoredChunk
    distance: float  # cosine distance: 0 = identical direction, 1 = unrelated, 2 = opposite

    @property
    def similarity(self) -> float:
        return 1.0 - self.distance


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str):
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self._get_or_create()

    def _get_or_create(self):
        # We compute embeddings ourselves (OpenAI), so Chroma only stores and searches them.
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop and recreate the collection so re-ingestion starts clean."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # collection may not exist yet
            pass
        self.collection = self._get_or_create()

    def add(self, chunks: Sequence[StoredChunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self.collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[list(e) for e in embeddings],
            metadatas=[
                {"source": c.source, "chunk_index": c.chunk_index, "start": c.start, "end": c.end}
                for c in chunks
            ],
        )

    def count(self) -> int:
        return self.collection.count()

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[SearchResult]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        # Chroma returns one list per query; we only send one query, hence [0].
        results: list[SearchResult] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunk = StoredChunk(
                id=chunk_id,
                text=text,
                source=str(meta["source"]),
                chunk_index=int(meta["chunk_index"]),
                start=int(meta["start"]),
                end=int(meta["end"]),
            )
            results.append(SearchResult(chunk=chunk, distance=float(distance)))
        return results
