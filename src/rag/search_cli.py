"""Try retrieval from the terminal.

  python -m src.rag.search_cli "What is the cancellation policy for Hotel Sunrise Goa?"
  python -m src.rag.search_cli            (interactive: type questions, empty line to quit)
  python -m src.rag.search_cli --rebuild  (re-index the documents first)
"""
import argparse
import sys
from pathlib import Path

from src.rag.chunker import chunk_documents
from src.rag.embedder import E5Embedder
from src.rag.loader import load_documents
from src.rag.vector_store import SearchResult, VectorStore

ROOT = Path(__file__).resolve().parents[2]
DOCS_FOLDER = ROOT / "data" / "travel_docs"
INDEX_FOLDER = ROOT / "data" / "chroma_db"  # ignored by git


def build_index(store: VectorStore, docs_folder: Path = DOCS_FOLDER) -> int:
    chunks = chunk_documents(load_documents(docs_folder))
    store.rebuild(chunks)
    return len(chunks)


def print_results(results: list[SearchResult]) -> None:
    if not results:
        print("  (no results)")
    for rank, r in enumerate(results, start=1):
        preview = r.text[:110].replace("\n", " ")
        print(f"  {rank}. score {r.score:.3f} | {r.chunk_id}\n     {preview}...")


def main() -> None:
    for stream in (sys.stdout, sys.stdin):  # so Hindi and Telugu text works on Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="a question to search for")
    parser.add_argument("--rebuild", action="store_true", help="re-index the documents")
    parser.add_argument("-k", type=int, default=3, help="how many results to show")
    args = parser.parse_args()

    store = VectorStore(E5Embedder(), INDEX_FOLDER)
    if args.rebuild or store.count() == 0:
        print("Indexing documents (the first run downloads the model, please wait)...")
        print(f"Indexed {build_index(store)} chunks.\n")

    if args.query:
        print_results(store.search(args.query, args.k))
        return
    while True:
        query = input("\nQuestion (empty line to quit): ").strip()
        if not query:
            break
        print_results(store.search(query, args.k))


if __name__ == "__main__":
    main()
