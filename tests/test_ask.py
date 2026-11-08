"""Tests for the CLI output formatting in ask.py."""

from ask import render
from docmind.pipeline import Answer
from docmind.vector_store import SearchResult, StoredChunk


def _result(chunk_id: str, distance: float) -> SearchResult:
    source, index = chunk_id.split("::")
    chunk = StoredChunk(id=chunk_id, text="some text", source=source,
                        chunk_index=int(index), start=0, end=9)
    return SearchResult(chunk=chunk, distance=distance)


def test_show_context_marks_chunks_filtered_by_threshold():
    kept, dropped = _result("a.md::0", 0.40), _result("b.md::3", 0.703)
    answer = Answer("Answer [S1].", sufficient_context=True, citations=[kept], retrieved=[kept, dropped])

    out = render(answer, show_context=True, max_distance=0.70)

    assert "a.md::0  similarity=0.600  distance=0.400  [sent to LLM]" in out
    assert "b.md::3  similarity=0.297  distance=0.703  [FILTERED (above max_distance)]" in out
    assert "[1] a.md (chunk 0, chars 0-9, similarity 0.60)" in out
