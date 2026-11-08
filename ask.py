"""CLI: ask a question about the indexed documents.

Usage:
    python ask.py "How many vacation days do employees get?"
    python ask.py "..." --top-k 6 --show-context
"""

from __future__ import annotations

import argparse
import sys

import openai

from docmind.config import ConfigError, load_settings
from docmind.factory import make_pipeline
from docmind.pipeline import Answer, EmptyIndexError


def render(answer: Answer, show_context: bool, max_distance: float) -> str:
    lines = ["", "Answer:", answer.text, ""]

    if answer.sufficient_context:
        lines.append("Sources:")
        if answer.citations:
            for i, r in enumerate(answer.citations, start=1):
                c = r.chunk
                lines.append(f"  [{i}] {c.source} (chunk {c.chunk_index}, chars {c.start}-{c.end}, "
                             f"similarity {r.similarity:.2f})")
        else:
            lines.append("  (the model did not cite any source - treat this answer with caution)")
    elif answer.retrieved:
        best = min(answer.retrieved, key=lambda r: r.distance)
        lines.append(f"(Closest chunk: {best.chunk.source}, similarity {best.similarity:.2f})")

    if show_context:
        lines += ["", f"Retrieved chunks (only those with distance <= {max_distance} are sent to the LLM):"]
        for r in answer.retrieved:
            c = r.chunk
            preview = c.text.replace("\n", " ")[:160]
            status = "sent to LLM" if r.distance <= max_distance else "FILTERED (above max_distance)"
            lines.append(f"  - {c.id}  similarity={r.similarity:.3f}  distance={r.distance:.3f}  [{status}]")
            lines.append(f"      {preview}...")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question about your indexed documents.")
    parser.add_argument("question", help="The question, in quotes")
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve")
    parser.add_argument("--show-context", action="store_true", help="Print all retrieved chunks and scores")
    args = parser.parse_args(argv)

    try:
        pipeline = make_pipeline(load_settings(), top_k=args.top_k)
        answer = pipeline.answer(args.question)
    except (ConfigError, EmptyIndexError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except openai.OpenAIError as exc:
        print(f"OpenAI API error: {exc}", file=sys.stderr)
        return 2

    print(render(answer, args.show_context, pipeline.max_distance))
    return 0


if __name__ == "__main__":
    sys.exit(main())
