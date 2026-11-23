from abc import ABC, abstractmethod


class LLMError(Exception):
    """Raised when a model call fails. Never put secrets in the message."""


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Return the model's text reply."""


class FakeLLMClient(LLMClient):
    """Predictable stand-in used by tests."""

    def generate(self, system: str, user: str) -> str:
        return f"[fake reply] {user}"


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        if not api_key or not model:
            raise ValueError("GEMINI_API_KEY and GEMINI_MODEL must be set in .env")
        # Imported here so tests and the fake client work without this library's setup.
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model = model

    def generate(self, system: str, user: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user,
                config=self._types.GenerateContentConfig(
                    system_instruction=system, temperature=0.2
                ),
            )
        except Exception as exc:
            raise LLMError("Gemini request failed") from exc
        text = response.text
        if not text:
            raise LLMError("Gemini returned an empty reply")
        return text


def get_llm_client(provider: str, api_key: str = "", model: str = "") -> LLMClient:
    if provider == "fake":
        return FakeLLMClient()
    if provider == "gemini":
        return GeminiClient(api_key, model)
    raise ValueError(f"Unknown LLM provider: {provider!r}")
