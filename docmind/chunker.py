"""Split text into overlapping chunks.

Strategy: fixed-size character windows with overlap, but each chunk tries to end
on a natural boundary (paragraph > line > sentence > word) so chunks don't cut
sentences in half. The overlap repeats the tail of one chunk at the start of the
next, so a fact that straddles a boundary still appears whole in at least one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

# Preferred break points, best first.
_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int  # character offset in the original document (inclusive)
    end: int  # character offset in the original document (exclusive)


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[Chunk] = []
    length = len(text)
    start = 0

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Look for a natural break in the second half of the window, so a
            # chunk is never shorter than half of chunk_size because of it.
            search_from = start + chunk_size // 2
            for sep in _SEPARATORS:
                idx = text.rfind(sep, search_from, end)
                if idx != -1:
                    end = idx + len(sep)
                    break

        piece = text[start:end]
        stripped = piece.strip()
        if stripped:
            # Record offsets of the stripped text so they point at real content.
            lead = len(piece) - len(piece.lstrip())
            chunks.append(Chunk(text=stripped, start=start + lead, end=start + lead + len(stripped)))

        if end >= length:
            break

        # Step back by `overlap`, then move forward to the next word start so
        # the overlapping chunk doesn't begin mid-word.
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        if space != -1 and overlap > 0:
            next_start = space + 1
        start = next_start

    return chunks
