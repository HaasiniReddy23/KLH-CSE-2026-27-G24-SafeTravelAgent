from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from src.config import get_settings
from src.llm_client import LLMClient, LLMError, get_llm_client
from src.rag.answerer import answer_question
from src.rag.embedder import E5Embedder
from src.rag.search_cli import INDEX_FOLDER, build_index
from src.rag.vector_store import VectorStore
from src.schemas import ChatRequest, ChatResponse, HealthResponse, RAGAnswer

SYSTEM_PROMPT = "You are a helpful travel assistant. Keep answers short and clear."

app = FastAPI(title="Safe Multilingual Travel Assistant")


@lru_cache
def get_llm() -> LLMClient:
    settings = get_settings()
    return get_llm_client(
        settings.llm_provider, settings.gemini_api_key, settings.gemini_model
    )


@lru_cache
def get_store() -> VectorStore:
    store = VectorStore(E5Embedder(), INDEX_FOLDER)
    if store.count() == 0:
        build_index(store)
    return store


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, llm: LLMClient = Depends(get_llm)) -> ChatResponse:
    try:
        reply = llm.generate(SYSTEM_PROMPT, request.message)
    except LLMError:
        raise HTTPException(status_code=502, detail="The language model is unavailable.")
    return ChatResponse(reply=reply)


@app.post("/ask", response_model=RAGAnswer)
def ask(
    request: ChatRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
) -> RAGAnswer:
    try:
        return answer_question(request.message, store, llm)
    except LLMError:
        raise HTTPException(status_code=502, detail="The language model is unavailable.")
