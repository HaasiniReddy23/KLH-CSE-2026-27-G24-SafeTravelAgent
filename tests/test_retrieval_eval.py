from pathlib import Path

from src.evaluation.retrieval_eval import (
    GoldenQuestion,
    evaluate_retrieval,
    hit_at_k,
    load_golden,
    ranked_sources,
    reciprocal_rank,
    summarize,
    threshold_table,
)
from src.rag.chunker import Chunk
from src.rag.embedder import FakeEmbedder
from src.rag.vector_store import SearchResult, VectorStore

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_FILE = ROOT / "data" / "eval" / "golden_questions.json"
DOCS_FOLDER = ROOT / "data" / "travel_docs"


def test_hit_at_k():
    assert hit_at_k(["a.txt"], ["b.txt", "a.txt"], 2) is True
    assert hit_at_k(["a.txt"], ["b.txt", "a.txt"], 1) is False
    assert hit_at_k(["a.txt"], [], 3) is False


def test_reciprocal_rank():
    assert reciprocal_rank(["a.txt"], ["a.txt", "b.txt"]) == 1.0
    assert reciprocal_rank(["a.txt"], ["b.txt", "a.txt"]) == 0.5
    assert reciprocal_rank(["a.txt"], ["b.txt", "c.txt"]) == 0.0


def test_ranked_sources_counts_each_document_once():
    results = [
        SearchResult("a#0", "a.txt", "x", 0.9),
        SearchResult("a#1", "a.txt", "y", 0.8),
        SearchResult("b#0", "b.txt", "z", 0.7),
    ]
    assert ranked_sources(results) == ["a.txt", "b.txt"]


# ---- the golden file itself ----
def test_golden_file_has_enough_varied_questions():
    questions = load_golden(GOLDEN_FILE)
    assert len(questions) >= 30
    assert {q.language for q in questions} == {"en", "hi", "te"}
    assert {q.category for q in questions} == {"answerable", "unanswerable"}
    assert len({q.id for q in questions}) == len(questions)


def test_golden_sources_point_to_real_documents():
    real = {p.name for p in DOCS_FOLDER.glob("*.txt")}
    for q in load_golden(GOLDEN_FILE):
        if q.category == "answerable":
            assert q.expected_sources, q.id
            assert set(q.expected_sources) <= real, q.id
        else:
            assert q.expected_sources == [], q.id


# ---- the evaluation with a small fake store ----
def make_store(tmp_path):
    store = VectorStore(FakeEmbedder(), tmp_path / "db")
    store.add_chunks(
        [
            Chunk("hotel.txt#0", "hotel.txt", 0, "hotel cancellation refund free cancel before check in"),
            Chunk("goa.txt#0", "goa.txt", 0, "beach monsoon season best time visit"),
        ]
    )
    return store


QUESTIONS = [
    GoldenQuestion("q1", "en", "answerable", "easy", "hotel cancellation refund", ["hotel.txt"], []),
    GoldenQuestion("q2", "en", "answerable", "easy", "beach monsoon", ["goa.txt"], []),
    GoldenQuestion("q3", "en", "unanswerable", "easy", "zebra giraffe", [], []),
]


def test_evaluate_and_summarize(tmp_path):
    rows = evaluate_retrieval(QUESTIONS, make_store(tmp_path))
    summary = summarize(rows)
    assert summary["all"]["questions"] == 2  # unanswerable questions are not counted
    assert summary["all"]["hit_at_1"] == 1.0
    assert summary["all"]["mrr"] == 1.0
    assert summary["en"]["questions"] == 2


def test_threshold_table(tmp_path):
    rows = evaluate_retrieval(QUESTIONS, make_store(tmp_path))
    table = threshold_table(rows, [0.5, 0.99])
    low, high = table
    assert low["answerable_wrongly_refused"] == 0
    assert low["unanswerable_wrongly_passed"] == 0
    assert high["answerable_wrongly_refused"] == 2  # a 0.99 cutoff refuses everything
