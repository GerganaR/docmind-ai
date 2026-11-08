"""End-to-end pipeline tests: real ChromaDB (temp folder) + fake embedder + fake LLM."""

import pytest

from docmind.pipeline import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    EmptyIndexError,
    RAGPipeline,
    extract_citations,
    ingest,
)
from docmind.prompts import INSUFFICIENT_CONTEXT_TOKEN, SYSTEM_PROMPT

from conftest import FakeChatModel


def _ingest(docs_dir, embedder, store):
    return ingest(docs_dir, embedder, store, chunk_size=120, chunk_overlap=20)


def test_ingest_stores_chunks_with_metadata(docs_dir, embedder, store):
    report = _ingest(docs_dir, embedder, store)

    assert report.documents == 2
    assert report.chunks == store.count() > 2
    results = store.search(embedder.embed(["vacation days"])[0], top_k=1)
    assert results[0].chunk.source == "handbook.md"
    assert results[0].chunk.id.startswith("handbook.md::")


def test_reingest_replaces_previous_index(docs_dir, embedder, store):
    _ingest(docs_dir, embedder, store)
    first = store.count()
    _ingest(docs_dir, embedder, store)
    assert store.count() == first  # no duplicates


def test_answer_includes_cited_sources(docs_dir, embedder, store):
    _ingest(docs_dir, embedder, store)
    llm = FakeChatModel("Full-time employees receive 28 vacation days per calendar year [S1].")
    pipeline = RAGPipeline(embedder, store, llm, top_k=3, max_distance=0.9)

    answer = pipeline.answer("How many vacation days do full-time employees receive?")

    assert answer.sufficient_context is True
    assert answer.citations[0].chunk.source == "handbook.md"
    assert "28 vacation days" in answer.citations[0].chunk.text
    system, user = llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "[S1] (file: handbook.md" in user
    assert "Question: How many vacation days" in user


def test_unrelated_question_returns_insufficient_without_calling_llm(docs_dir, embedder, store):
    _ingest(docs_dir, embedder, store)
    llm = FakeChatModel()
    pipeline = RAGPipeline(embedder, store, llm, top_k=3, max_distance=0.5)

    answer = pipeline.answer("Which pizza toppings does the CEO prefer?")

    assert answer.sufficient_context is False
    assert answer.text == INSUFFICIENT_CONTEXT_MESSAGE
    assert answer.citations == []
    assert llm.calls == []  # guardrail 1: no LLM cost for irrelevant questions


def test_llm_insufficient_token_is_respected(docs_dir, embedder, store):
    _ingest(docs_dir, embedder, store)
    llm = FakeChatModel(INSUFFICIENT_CONTEXT_TOKEN)
    pipeline = RAGPipeline(embedder, store, llm, top_k=3, max_distance=2.0)

    answer = pipeline.answer("How many vacation days do interns get?")

    assert answer.sufficient_context is False
    assert answer.text == INSUFFICIENT_CONTEXT_MESSAGE
    assert len(llm.calls) == 1  # guardrail 2: the LLM made the call


def test_empty_index_raises(embedder, store):
    pipeline = RAGPipeline(embedder, store, FakeChatModel())
    with pytest.raises(EmptyIndexError):
        pipeline.answer("Anything?")


def test_blank_question_raises(docs_dir, embedder, store):
    _ingest(docs_dir, embedder, store)
    with pytest.raises(ValueError):
        RAGPipeline(embedder, store, FakeChatModel()).answer("   ")


def test_extract_citations_ignores_unknown_and_duplicate_labels(docs_dir, embedder, store):
    _ingest(docs_dir, embedder, store)
    results = store.search(embedder.embed(["vacation"])[0], top_k=2)

    cited = extract_citations("A [S2]. B [S2][S1]. C [S9].", results)

    assert cited == [results[1], results[0]]
