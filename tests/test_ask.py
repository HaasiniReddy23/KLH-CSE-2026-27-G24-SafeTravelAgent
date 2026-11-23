import json

from fastapi.testclient import TestClient

from src.llm_client import LLMClient, LLMError
from src.main import app, get_llm, get_store
from src.rag.chunker import Chunk
from src.rag.embedder import FakeEmbedder
from src.rag.vector_store import VectorStore

GOOD = json.dumps(
    {"found": True, "answer": "Free cancellation until 48 hours before.",
        "sources": ["hotel.txt#0"]}
)


class FixedLLM(LLMClient):
    def __init__(self, reply: str):
        self.reply = reply

    def generate(self, system: str, user: str) -> str:
        return self.reply


class BrokenLLM(LLMClient):
    def generate(self, system: str, user: str) -> str:
        raise LLMError("secret-internal-details")


def make_client(llm, tmp_path) -> TestClient:
    store = VectorStore(FakeEmbedder(), tmp_path / "db")
    store.add_chunks(
        [Chunk("hotel.txt#0", "hotel.txt", 0, "hotel cancellation refund")]
    )
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_ask_returns_cited_answer(tmp_path):
    client = make_client(FixedLLM(GOOD), tmp_path)
    response = client.post(
        "/ask", json={"message": "hotel cancellation refund"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "answered"
    assert body["citations"][0]["chunk_id"] == "hotel.txt#0"


def test_ask_rejects_empty_message(tmp_path):
    client = make_client(FixedLLM(GOOD), tmp_path)
    assert client.post("/ask", json={"message": ""}).status_code == 422


def test_ask_hides_internal_llm_errors(tmp_path):
    client = make_client(BrokenLLM(), tmp_path)
    response = client.post(
        "/ask", json={"message": "hotel cancellation refund"})
    assert response.status_code == 502
    assert "secret-internal-details" not in response.text
