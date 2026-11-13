"""Retrieval evaluation.

Each question in questions.json names the document that holds the answer and
a phrase the right chunk must contain. A retrieved chunk counts as correct
only if both match, so landing in the right file but the wrong section is a
miss.

Metrics
  hit@k : share of questions with a correct chunk in the top k
  MRR   : mean of 1 / rank of the first correct chunk (0 if not in top 10)
"""

import json
from pathlib import Path

QUESTIONS = Path(__file__).parent / "questions.json"


def is_correct(hit, item: dict) -> bool:
    return hit.chunk.source == item["source"] and item["must_contain"].lower() in hit.chunk.text.lower()


def first_correct_rank(index, item: dict, mode: str, depth: int = 10):
    for rank, hit in enumerate(index.search(item["question"], depth, mode), 1):
        if is_correct(hit, item):
            return rank
    return None


def evaluate(index, questions: list = None, modes=("bm25", "dense", "hybrid")) -> dict:
    questions = questions or json.loads(QUESTIONS.read_text(encoding="utf-8"))
    results = {}
    for mode in modes:
        ranks = [first_correct_rank(index, q, mode) for q in questions]
        n = len(ranks)
        results[mode] = {
            "hit@1": sum(1 for r in ranks if r and r <= 1) / n,
            "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
            "hit@5": sum(1 for r in ranks if r and r <= 5) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
            "misses": [q["question"] for q, r in zip(questions, ranks) if not r or r > 3],
        }
    return results


def format_table(results: dict, n_questions: int, embedder: str) -> str:
    lines = [
        f"Retrieval evaluation: {n_questions} questions, embedder = {embedder}",
        "",
        "| Mode   | hit@1 | hit@3 | hit@5 | MRR   |",
        "|--------|-------|-------|-------|-------|",
    ]
    for mode, r in results.items():
        lines.append(
            f"| {mode:<6} | {r['hit@1']:.2f}  | {r['hit@3']:.2f}  | {r['hit@5']:.2f}  | {r['mrr']:.3f} |"
        )
    return "\n".join(lines)
