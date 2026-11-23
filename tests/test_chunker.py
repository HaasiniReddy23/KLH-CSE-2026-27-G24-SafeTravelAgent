import pytest

from src.rag.chunker import chunk_documents, chunk_text
from src.rag.loader import Document


def make_text(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_text("one two three", max_words=10, overlap_words=2) == ["one two three"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_chunks_overlap_by_the_requested_words():
    chunks = chunk_text(make_text(10), max_words=4, overlap_words=1)
    assert len(chunks) == 3
    assert chunks[0].split()[-1] == chunks[1].split()[0]  # shared word
    assert chunks[1].split()[-1] == chunks[2].split()[0]


def test_no_words_are_lost():
    text = make_text(37)
    chunks = chunk_text(text, max_words=10, overlap_words=3)
    seen = set(word for chunk in chunks for word in chunk.split())
    assert seen == set(text.split())


def test_no_chunk_is_longer_than_max_words():
    for chunk in chunk_text(make_text(100), max_words=12, overlap_words=4):
        assert len(chunk.split()) <= 12


def test_invalid_settings_raise_errors():
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=5, overlap_words=5)
    with pytest.raises(ValueError):
        chunk_text("a b c", max_words=0, overlap_words=0)


def test_chunk_documents_keeps_source_and_ids():
    docs = [Document(source="x.txt", text=make_text(10))]
    chunks = chunk_documents(docs, max_words=4, overlap_words=1)
    assert [c.chunk_id for c in chunks] == ["x.txt#0", "x.txt#1", "x.txt#2"]
    assert all(c.source == "x.txt" for c in chunks)
