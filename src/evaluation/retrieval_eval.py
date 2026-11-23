"""Measure retrieval quality on the golden question set.

Hit Rate@k : share of answerable questions whose correct document is in the top k.
MRR        : mean of 1/rank of the first correct document (1.0 = always first).
Both are measured at DOCUMENT level, so changing chunk size does not break them.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from src.rag.vector_store import SearchResult, VectorStore


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    language: str  # "en", "hi" or "te"
    category: str  # "answerable" or "unanswerable"
    difficulty: str
    question: str
    expected_sources: list[str]
    expected_facts: list[str]
    note: str = ""


@dataclass(frozen=True)
class QuestionResult:
    id: str
    language: str
    category: str
    question: str
    expected_sources: list[str]
    top_score: float | None
    ranked_sources: list[str]
    hit_at_1: bool
    hit_at_3: bool
    reciprocal_rank: float


def load_golden(path: Path) -> list[GoldenQuestion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenQuestion(**item) for item in raw]


def ranked_sources(results: list[SearchResult]) -> list[str]:
    """Document names in rank order, each document counted once."""
    seen: list[str] = []
    for result in results:
        if result.source not in seen:
            seen.append(result.source)
    return seen


def hit_at_k(expected: list[str], ranked: list[str], k: int) -> bool:
    return any(source in expected for source in ranked[:k])


def reciprocal_rank(expected: list[str], ranked: list[str]) -> float:
    for position, source in enumerate(ranked, start=1):
        if source in expected:
            return 1.0 / position
    return 0.0


def evaluate_retrieval(
    questions: list[GoldenQuestion], store: VectorStore, top_k: int = 5
) -> list[QuestionResult]:
    rows = []
    for q in questions:
        results = store.search(q.question, top_k)
        ranked = ranked_sources(results)
        answerable = q.category == "answerable"
        rows.append(
            QuestionResult(
                id=q.id,
                language=q.language,
                category=q.category,
                question=q.question,
                expected_sources=q.expected_sources,
                top_score=results[0].score if results else None,
                ranked_sources=ranked,
                hit_at_1=answerable and hit_at_k(q.expected_sources, ranked, 1),
                hit_at_3=answerable and hit_at_k(q.expected_sources, ranked, 3),
                reciprocal_rank=reciprocal_rank(q.expected_sources, ranked) if answerable else 0.0,
            )
        )
    return rows


def _group_summary(rows: list[QuestionResult]) -> dict:
    count = len(rows)
    if count == 0:
        return {"questions": 0, "hit_at_1": None, "hit_at_3": None, "mrr": None}
    return {
        "questions": count,
        "hit_at_1": round(sum(r.hit_at_1 for r in rows) / count, 3),
        "hit_at_3": round(sum(r.hit_at_3 for r in rows) / count, 3),
        "mrr": round(sum(r.reciprocal_rank for r in rows) / count, 3),
    }


def summarize(rows: list[QuestionResult]) -> dict:
    """Retrieval metrics for answerable questions, overall and per language."""
    answerable = [r for r in rows if r.category == "answerable"]
    summary = {"all": _group_summary(answerable)}
    for language in sorted({r.language for r in answerable}):
        summary[language] = _group_summary([r for r in answerable if r.language == language])
    return summary


def threshold_table(rows: list[QuestionResult], thresholds: list[float]) -> list[dict]:
    """For each cutoff: how many answerable questions would be refused wrongly,
    and how many unanswerable ones would still pass the score gate."""
    answerable = [r for r in rows if r.category == "answerable" and r.top_score is not None]
    unanswerable = [r for r in rows if r.category == "unanswerable"]
    table = []
    for t in thresholds:
        table.append(
            {
                "threshold": round(t, 2),
                "answerable_wrongly_refused": sum(r.top_score < t for r in answerable),
                "answerable_total": len(answerable),
                "unanswerable_wrongly_passed": sum(
                    r.top_score is not None and r.top_score >= t for r in unanswerable
                ),
                "unanswerable_total": len(unanswerable),
            }
        )
    return table
