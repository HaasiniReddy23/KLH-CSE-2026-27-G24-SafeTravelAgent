from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from src.config import get_settings
from src.llm_client import LLMClient, LLMError, get_llm_client
from src.schemas import ChatRequest, ChatResponse, HealthResponse

SYSTEM_PROMPT = "You are a helpful travel assistant. Keep answers short and clear."

app = FastAPI(title="Safe Multilingual Travel Assistant")


@lru_cache
def get_llm() -> LLMClient:
    settings = get_settings()
    return get_llm_client(
        settings.llm_provider, settings.gemini_api_key, settings.gemini_model
    )


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
