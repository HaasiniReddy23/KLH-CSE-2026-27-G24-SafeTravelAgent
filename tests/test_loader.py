from src.rag.loader import clean_text, load_documents


def test_clean_text_removes_extra_spaces_and_blank_lines():
    assert clean_text("a   b\n\n\n\nc  ") == "a b\n\nc"


def test_load_documents_reads_only_txt_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "b.txt").write_text("second doc", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("not loaded", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.source for d in docs] == ["a.txt", "b.txt"]


def test_load_documents_skips_empty_files(tmp_path):
    (tmp_path / "empty.txt").write_text("   \n  ", encoding="utf-8")
    assert load_documents(tmp_path) == []


def test_load_documents_handles_hindi_and_telugu(tmp_path):
    (tmp_path / "hi.txt").write_text("गोवा में होटल", encoding="utf-8")
    (tmp_path / "te.txt").write_text("గోవాలో హోటల్స్", encoding="utf-8")
    texts = [d.text for d in load_documents(tmp_path)]
    assert "गोवा में होटल" in texts and "గోవాలో హోటల్స్" in texts
