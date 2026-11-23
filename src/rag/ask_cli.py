"""Ask a full question from the terminal (needs LLM_PROVIDER=gemini in .env).

  python -m src.rag.ask_cli "What is the cancellation policy for Hotel Sunrise Goa?"
  python -m src.rag.ask_cli            (interactive; empty line to quit)
"""
import sys

from src.config import get_settings
from src.llm_client import get_llm_client
from src.rag.answerer import answer_question
from src.rag.embedder import E5Embedder
from src.rag.search_cli import INDEX_FOLDER, build_index
from src.rag.vector_store import VectorStore
from src.schemas import RAGAnswer


def show(result: RAGAnswer) -> None:
    print(f"\n[{result.status}] reason={result.reason} top_score={result.top_score}")
    print(result.answer)
    for c in result.citations:
        print(f"  source: {c.chunk_id}")


def main() -> None:
    for stream in (sys.stdout, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    settings = get_settings()
    llm = get_llm_client(
        settings.llm_provider, settings.gemini_api_key, settings.gemini_model
    )
    store = VectorStore(E5Embedder(), INDEX_FOLDER)
    if store.count() == 0:
        print(f"Indexed {build_index(store)} chunks.")

    if len(sys.argv) > 1:
        show(answer_question(" ".join(sys.argv[1:]), store, llm))
        return
    while True:
        question = input("\nQuestion (empty line to quit): ").strip()
        if not question:
            break
        show(answer_question(question, store, llm))


if __name__ == "__main__":
    main()
