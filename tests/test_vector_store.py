from src.rag.chunker import Chunk
from src.rag.embedder import FakeEmbedder
from src.rag.vector_store import VectorStore

CHUNKS = [
    Chunk("hotel.txt#0", "hotel.txt", 0, "hotel cancellation refund free cancel before check in"),
    Chunk("flight.txt#0", "flight.txt", 0, "flight baggage kilograms cabin checked"),
    Chunk("goa.txt#0", "goa.txt", 0, "beach monsoon season best time visit"),
]


def make_store(tmp_path):
    return VectorStore(FakeEmbedder(), tmp_path / "db")


def test_search_returns_the_relevant_chunk_first(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks(CHUNKS)
    results = store.search("hotel cancellation refund", k=2)
    assert results[0].chunk_id == "hotel.txt#0"
    assert results[0].score > results[1].score


def test_results_carry_source_and_text(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks(CHUNKS)
    top = store.search("flight baggage", k=1)[0]
    assert top.source == "flight.txt"
    assert "baggage" in top.text


def test_k_larger_than_store_returns_everything(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks(CHUNKS)
    assert len(store.search("hotel", k=10)) == 3


def test_empty_store_returns_nothing(tmp_path):
    assert make_store(tmp_path).search("hotel") == []


def test_blank_query_returns_nothing(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks(CHUNKS)
    assert store.search("   ") == []


def test_rebuild_replaces_old_chunks(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks(CHUNKS)
    store.rebuild([CHUNKS[1]])
    assert store.count() == 1
    assert store.search("hotel", k=5)[0].chunk_id == "flight.txt#0"


def test_unrelated_query_scores_near_zero(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks([CHUNKS[0]])
    assert abs(store.search("zebra giraffe", k=1)[0].score) < 0.1


def test_index_survives_reopening(tmp_path):
    make_store(tmp_path).add_chunks(CHUNKS)
    assert make_store(tmp_path).count() == 3
