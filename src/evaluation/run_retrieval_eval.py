"""Run the retrieval evaluation with the real model:

  python -m src.evaluation.run_retrieval_eval

Prints the tables and saves them to results/retrieval_eval.json (commit that file).
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

from src.evaluation.retrieval_eval import (
    evaluate_retrieval,
    load_golden,
    summarize,
    threshold_table,
)
from src.rag.embedder import E5Embedder
from src.rag.search_cli import INDEX_FOLDER, build_index
from src.rag.vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FILE = ROOT / "data" / "eval" / "golden_questions.json"
RESULTS_FILE = ROOT / "results" / "retrieval_eval.json"
THRESHOLDS = [0.70 + 0.01 * i for i in range(0, 21)]  # 0.70 ... 0.90


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    questions = load_golden(GOLDEN_FILE)
    store = VectorStore(E5Embedder(), INDEX_FOLDER)
    print(f"Re-indexing documents: {build_index(store)} chunks")  # fresh, matches the files
    rows = evaluate_retrieval(questions, store)

    summary = summarize(rows)
    print("\nRETRIEVAL (answerable questions only)")
    print(f"{'group':<6}{'questions':>10}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>8}")
    for group, s in summary.items():
        print(f"{group:<6}{s['questions']:>10}{s['hit_at_1']:>8}{s['hit_at_3']:>8}{s['mrr']:>8}")

    print("\nMISSES (answerable, correct document not ranked first)")
    misses = [r for r in rows if r.category == "answerable" and not r.hit_at_1]
    for r in misses:
        print(f"  {r.id} [{r.language}] score={r.top_score:.3f} expected={r.expected_sources} got={r.ranked_sources[:3]}")
    if not misses:
        print("  none")

    print("\nTOP SCORES")
    for category in ("answerable", "unanswerable"):
        scores = sorted(r.top_score for r in rows if r.category == category and r.top_score is not None)
        print(f"  {category:<13} min={scores[0]:.3f}  max={scores[-1]:.3f}  count={len(scores)}")

    table = threshold_table(rows, THRESHOLDS)
    print("\nSCORE CUTOFF (the score gate alone, before the LLM)")
    print(f"{'cutoff':>7}  {'answerable wrongly refused':>27}  {'unanswerable wrongly passed':>28}")
    for t in table:
        print(
            f"{t['threshold']:>7}  "
            f"{t['answerable_wrongly_refused']:>10} of {t['answerable_total']:<14}  "
            f"{t['unanswerable_wrongly_passed']:>11} of {t['unanswerable_total']:<14}"
        )

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(
        json.dumps(
            {"summary": summary, "thresholds": table, "rows": [asdict(r) for r in rows]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {RESULTS_FILE}")


if __name__ == "__main__":
    main()
