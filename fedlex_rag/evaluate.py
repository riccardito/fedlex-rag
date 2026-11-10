"""Retrieval metrics on the evaluation set: Recall@k, hit@1 and MRR@10."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .loader import DATA

QUESTIONS = DATA / "questions.jsonl"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    expected: list[str]   # empty: the sources do not answer it (the refusal probes)


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    with open(path, encoding="utf-8") as fh:
        return [Question(**json.loads(line)) for line in fh if line.strip()]


def recall_at(ranked: list[str], expected: list[str], k: int) -> float:
    """The share of the expected articles among the first k."""
    return len(set(ranked[:k]) & set(expected)) / len(expected)


def reciprocal_rank(ranked: list[str], expected: list[str], k: int = 10) -> float:
    for i, r in enumerate(ranked[:k], start=1):
        if r in expected:
            return 1 / i
    return 0.0


@dataclass(frozen=True)
class Scores:
    name: str
    n: int
    recall: float
    hit1: float
    mrr: float
    misses: list[str]


def evaluate(retriever, questions: list[Question], k: int = 5) -> Scores:
    answerable = [q for q in questions if q.expected]
    recall = hit1 = mrr = 0.0
    misses = []
    for q in answerable:
        ranked = [a.id for a, _ in retriever.search(q.question, k=max(k, 10))]
        r = recall_at(ranked, q.expected, k)
        recall += r
        hit1 += recall_at(ranked, q.expected, 1) > 0
        mrr += reciprocal_rank(ranked, q.expected)
        if r < 1:
            misses.append(q.id)
    n = len(answerable)
    return Scores(retriever.name, n, recall / n, hit1 / n, mrr / n, misses)
