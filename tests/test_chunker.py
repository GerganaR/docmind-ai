import pytest

from docmind.chunker import split_text


def test_empty_text_returns_no_chunks():
    assert split_text("   ", chunk_size=100, overlap=10) == []


def test_short_text_is_single_chunk():
    chunks = split_text("Hello world.", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert (chunks[0].start, chunks[0].end) == (0, 12)


def test_chunks_respect_size_and_cover_all_words():
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words)
    chunks = split_text(text, chunk_size=200, overlap=40)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    covered = set(" ".join(c.text for c in chunks).split())
    assert covered == set(words)  # nothing lost


def test_consecutive_chunks_overlap():
    text = " ".join(f"token{i}" for i in range(300))
    chunks = split_text(text, chunk_size=150, overlap=50)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start < prev.end  # windows overlap in the source text
        assert prev.text.split()[-1] in nxt.text  # boundary word repeated


def test_offsets_point_to_original_text():
    text = "First paragraph sentence one. Sentence two.\n\nSecond paragraph here. " * 10
    for c in split_text(text, chunk_size=120, overlap=30):
        assert text[c.start:c.end] == c.text


def test_prefers_paragraph_boundaries():
    para = "A" * 60
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = split_text(text, chunk_size=100, overlap=0)
    assert chunks[0].text == para


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=size, overlap=overlap)
