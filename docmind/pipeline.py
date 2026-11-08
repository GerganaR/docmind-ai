"""The RAG pipeline: ingestion (load -> chunk -> embed -> store) and question answering
(embed question -> retrieve -> threshold -> prompt -> LLM -> cited answer)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from docmind.chunker import split_text
from docmind.llm import ChatModel, Embedder
from docmind.loader import load_documents
from docmind.prompts import INSUFFICIENT_CONTEXT_TOKEN, SYSTEM_PROMPT, build_user_prompt
from docmind.vector_store import SearchResult, StoredChunk, VectorStore

INSUFFICIENT_CONTEXT_MESSAGE = (
    "I could not find enough information in the indexed documents to answer this question."
)

_CITATION_RE = re.compile(r"\[S(\d+)\]")


class EmptyIndexError(Exception):
    """Raised when asking a question before any documents were ingested."""


@dataclass
class IngestReport:
    documents: int
    chunks: int


@dataclass
class Answer:
    text: str
    sufficient_context: bool
    # Sources the answer cites (what we show the user).
    citations: list[SearchResult] = field(default_factory=list)
    # Everything retrieved, including chunks filtered out by the distance threshold (for debugging).
    retrieved: list[SearchResult] = field(default_factory=list)


def ingest(docs_dir: Path, embedder: Embedder, store: VectorStore,
           chunk_size: int, chunk_overlap: int) -> IngestReport:
    documents = load_documents(docs_dir)
    chunks: list[StoredChunk] = []
    for doc in documents:
        for index, piece in enumerate(split_text(doc.text, chunk_size, chunk_overlap)):
            chunks.append(
                StoredChunk(
                    id=f"{doc.source}::{index}",
                    text=piece.text,
                    source=doc.source,
                    chunk_index=index,
                    start=piece.start,
                    end=piece.end,
                )
            )

    store.reset()
    if chunks:
        embeddings = embedder.embed([c.text for c in chunks])
        store.add(chunks, embeddings)
    return IngestReport(documents=len(documents), chunks=len(chunks))


def extract_citations(answer_text: str, results: list[SearchResult]) -> list[SearchResult]:
    """Map [S#] labels in the answer back to the retrieved chunks, in order of first use."""
    cited: list[SearchResult] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(answer_text):
        n = int(match.group(1))
        if 1 <= n <= len(results) and n not in seen:  # ignore hallucinated labels like [S9]
            seen.add(n)
            cited.append(results[n - 1])
    return cited


class RAGPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, llm: ChatModel,
                 top_k: int = 4, max_distance: float = 0.70):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.max_distance = max_distance

    def retrieve(self, question: str) -> list[SearchResult]:
        if self.store.count() == 0:
            raise EmptyIndexError("The index is empty. Run `python ingest.py` first.")
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, self.top_k)

    def answer(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        retrieved = self.retrieve(question)
        relevant = [r for r in retrieved if r.distance <= self.max_distance]

        # Guardrail 1: nothing similar enough -> don't call the LLM at all.
        if not relevant:
            return Answer(INSUFFICIENT_CONTEXT_MESSAGE, sufficient_context=False, retrieved=retrieved)

        raw = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, relevant))

        # Guardrail 2: the LLM itself decides the context does not answer the question.
        if not raw or raw.strip().startswith(INSUFFICIENT_CONTEXT_TOKEN):
            return Answer(INSUFFICIENT_CONTEXT_MESSAGE, sufficient_context=False, retrieved=retrieved)

        return Answer(
            text=raw,
            sufficient_context=True,
            citations=extract_citations(raw, relevant),
            retrieved=retrieved,
        )
