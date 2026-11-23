from dataclasses import dataclass

from src.rag.loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # for example "hotel_sunrise_goa_policy.txt#0"
    source: str
    position: int  # 0 for the first chunk of a document, 1 for the next, ...
    text: str


def chunk_text(text: str, max_words: int = 120, overlap_words: int = 30) -> list[str]:
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    words = text.split()
    if not words:
        return []

    step = max_words - overlap_words
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):  # this piece reached the end
            break
    return pieces


def chunk_documents(
    documents: list[Document], max_words: int = 120, overlap_words: int = 30
) -> list[Chunk]:
    chunks = []
    for doc in documents:
        for position, piece in enumerate(chunk_text(doc.text, max_words, overlap_words)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}#{position}",
                    source=doc.source,
                    position=position,
                    text=piece,
                )
            )
    return chunks
