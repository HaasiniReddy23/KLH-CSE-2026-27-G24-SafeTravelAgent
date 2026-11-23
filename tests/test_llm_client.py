import pytest

from src.llm_client import FakeLLMClient, GeminiClient, get_llm_client


def test_fake_client_is_predictable():
    assert FakeLLMClient().generate("sys", "hi") == "[fake reply] hi"


def test_factory_returns_fake_client():
    assert isinstance(get_llm_client("fake"), FakeLLMClient)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        get_llm_client("banana")


def test_gemini_client_requires_key_and_model():
    with pytest.raises(ValueError):
        GeminiClient(api_key="", model="")
