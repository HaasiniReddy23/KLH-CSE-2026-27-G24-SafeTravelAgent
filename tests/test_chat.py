from fastapi.testclient import TestClient

from src.llm_client import FakeLLMClient, LLMClient, LLMError
from src.main import app, get_llm


class BrokenLLM(LLMClient):
    def generate(self, system: str, user: str) -> str:
        raise LLMError("secret-internal-details")


def make_client(llm: LLMClient) -> TestClient:
    app.dependency_overrides[get_llm] = lambda: llm
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_chat_returns_reply():
    client = make_client(FakeLLMClient())
    response = client.post("/chat", json={"message": "hello Goa"})
    assert response.status_code == 200
    assert "hello Goa" in response.json()["reply"]


def test_chat_rejects_empty_message():
    client = make_client(FakeLLMClient())
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_too_long_message():
    client = make_client(FakeLLMClient())
    assert client.post("/chat", json={"message": "a" * 2001}).status_code == 422


def test_chat_hides_internal_error_details():
    client = make_client(BrokenLLM())
    response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert "secret-internal-details" not in response.text
