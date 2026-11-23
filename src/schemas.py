from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str


class HealthResponse(BaseModel):
    status: str


class Citation(BaseModel):
    chunk_id: str  # e.g. "hotel_sunrise_goa_policy.txt#0"
    source: str  # the document file name


class RAGAnswer(BaseModel):
    status: Literal["answered", "no_answer"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    top_score: float | None = None  # best retrieval score, kept for logging and evaluation
    reason: str  # why we answered or refused, e.g. "low_retrieval_score"
