"""CLI: index the documents folder into ChromaDB.

Usage:
    python ingest.py
    python ingest.py --docs-dir path/to/docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openai

from docmind.config import ConfigError, load_settings
from docmind.factory import make_embedder, make_openai_client, make_store
from docmind.pipeline import ingest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index .txt/.md documents into ChromaDB.")
    parser.add_argument("--docs-dir", type=Path, help="Folder with documents (default: data/sample_docs)")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        docs_dir = args.docs_dir or settings.docs_dir
        client = make_openai_client(settings)
        report = ingest(
            docs_dir=docs_dir,
            embedder=make_embedder(settings, client),
            store=make_store(settings),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except openai.OpenAIError as exc:
        print(f"OpenAI API error: {exc}", file=sys.stderr)
        return 2

    if report.documents == 0:
        print(f"No .txt or .md documents found in {docs_dir}. The index is now empty.")
        return 1

    print(f"Indexed {report.documents} documents as {report.chunks} chunks "
          f"(chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}) "
          f"into {settings.chroma_dir}")
    return 0


# Python note: this block runs only when the file is executed directly
# (`python ingest.py`), not when it is imported by another module or a test.
if __name__ == "__main__":
    sys.exit(main())
