from pathlib import Path

from src.rag.chunker import chunk_documents
from src.rag.loader import load_documents

DOCS_FOLDER = Path(__file__).resolve().parents[2] / "data" / "travel_docs"


def main() -> None:
    documents = load_documents(DOCS_FOLDER)
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents)} documents -> {len(chunks)} chunks\n")
    for chunk in chunks:
        preview = chunk.text[:70].replace("\n", " ")
        print(f"{chunk.chunk_id:<45} {len(chunk.text.split()):>3} words | {preview}...")


if __name__ == "__main__":
    main()
