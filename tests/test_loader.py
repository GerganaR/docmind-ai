import pytest

from docmind.loader import load_documents


def test_loads_txt_and_md_only_and_skips_empty(tmp_path):
    (tmp_path / "a.md").write_text("# Title\nBody", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "c.md").write_text("nested doc", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert [d.source for d in docs] == ["a.md", "b.txt", "nested/c.md"]
    assert docs[1].text == "plain text"


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")
