"""Load plain-text and Markdown documents from a local folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass(frozen=True)
class Document:
    source: str  # path relative to the docs folder, e.g. "employee_handbook.md"
    text: str


def load_documents(folder: Path) -> list[Document]:
    """Recursively read every .txt/.md file under ``folder``.

    Empty files are skipped. Files are returned in a stable (sorted) order so
    ingestion is deterministic.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {folder}")

    documents: list[Document] = []
    # Path.rglob("*") walks the folder tree, like a recursive fs.readdir.
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(Document(source=path.relative_to(folder).as_posix(), text=text))
    return documents
