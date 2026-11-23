from src.rag.embedder import E5Embedder, FakeEmbedder


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_fake_embedder_is_deterministic():
    e = FakeEmbedder()
    assert e.embed_query("hotel refund") == e.embed_query("hotel refund")


def test_fake_embedder_vectors_have_length_one():
    vector = FakeEmbedder().embed_query("hotel cancellation refund")
    assert abs(dot(vector, vector) - 1.0) < 1e-9


def test_similar_text_scores_higher_than_unrelated_text():
    e = FakeEmbedder()
    query = e.embed_query("hotel cancellation refund")
    close = e.embed_passages(["hotel cancellation policy and refund rules"])[0]
    far = e.embed_passages(["scooter rental driving licence"])[0]
    assert dot(query, close) > dot(query, far)


def test_e5_prefixes_are_added():
    assert E5Embedder.format_query("hi") == "query: hi"
    assert E5Embedder.format_passage("hi") == "passage: hi"


def test_e5_model_is_not_loaded_until_needed():
    assert E5Embedder()._model is None


def test_e5_empty_passage_list_returns_empty_list():
    assert E5Embedder().embed_passages([]) == []
