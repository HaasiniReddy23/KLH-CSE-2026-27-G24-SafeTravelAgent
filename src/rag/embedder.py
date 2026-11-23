"""Turns text into lists of numbers ("embeddings") that capture meaning."""
import hashlib
import math
import re
from abc import ABC, abstractmethod

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class Embedder(ABC):
    @abstractmethod
    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a user question."""


class E5Embedder(Embedder):
    """Real multilingual model. E5 needs 'query: ' / 'passage: ' in front of text."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None  # loaded on first use (the download happens once)

    @staticmethod
    def format_passage(text: str) -> str:
        return f"passage: {text}"

    @staticmethod
    def format_query(text: str) -> str:
        return f"query: {text}"

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._get_model().encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [vector.tolist() for vector in vectors]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([self.format_passage(t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.format_query(text)])[0]


class FakeEmbedder(Embedder):
    """Word-counting stand-in for tests: no download, same answer every time.
    It only matches shared words, so it is NOT multilingual or semantic."""

    def __init__(self, dimensions: int = 256):
        self._dim = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for word in re.findall(r"\w+", text.lower()):
            index = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self._dim
            vector[index] += 1.0
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [x / norm for x in vector]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
