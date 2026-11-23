"""Chroma vector store: saves chunks with their embeddings and finds the closest ones."""
from dataclasses import dataclass
from pathlib import Path

from src.rag.chunker import Chunk
from src.rag.embedder import Embedder

COLLECTION_NAME = "travel_docs"


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    source: str
    text: str
    score: float  # cosine similarity: about 1.0 = very close, about 0 = unrelated


class VectorStore:
    def __init__(
        self, embedder: Embedder, persist_dir: Path, collection_name: str = COLLECTION_NAME
    ):
        import chromadb  # imported here so other modules load without it

        self._embedder = embedder
        self._name = collection_name
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._open_collection()

    def _open_collection(self):
        return self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=self._embedder.embed_passages([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "position": c.position} for c in chunks],
        )

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Throw away everything stored and index these chunks (no stale leftovers)."""
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass  # nothing to delete the first time
        self._collection = self._open_collection()
        self.add_chunks(chunks)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        total = self.count()
        if total == 0 or not query.strip():
            return []
        k = max(1, min(k, total))
        raw = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for chunk_id, text, meta, distance in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    source=meta["source"],
                    text=text,
                    score=1.0 - float(distance),
                )
            )
        return results
