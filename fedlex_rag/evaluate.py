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


def pool_recall(pipeline, questions: list[Question]) -> float:
    """How often an expected article is among the rerank's candidates at all -- the rerank's ceiling."""
    answerable = [q for q in questions if q.expected]
    hit = sum(bool({a.id for a in pipeline.candidates(q.question)} & set(q.expected)) for q in answerable)
    return hit / len(answerable)


@dataclass(frozen=True)
class AnswerScores:
    n_answerable: int
    n_unanswerable: int
    cited: float           # answerable questions answered AND citing an expected article
    false_refusal: float   # answerable questions refused
    refusal: float         # unanswerable questions correctly refused
    unsupported: int       # citations of an article that was not in the context
    misses: list[str]


def evaluate_answers(pipeline, questions: list[Question]) -> AnswerScores:
    ok = refused_answerable = refused_unanswerable = unsupported = 0
    misses = []
    for q in questions:
        _, answer = pipeline.ask(q.question)
        unsupported += len(answer.unsupported)
        if q.expected:
            good = answer.answerable and bool(set(answer.citations) & set(q.expected))
            ok += good
            refused_answerable += not answer.answerable
            if not good:
                misses.append(q.id)
        else:
            refused_unanswerable += not answer.answerable
            if answer.answerable:
                misses.append(q.id)
    n_a = sum(bool(q.expected) for q in questions)
    n_u = len(questions) - n_a
    return AnswerScores(n_a, n_u, ok / n_a, refused_answerable / n_a,
                        refused_unanswerable / n_u if n_u else float("nan"), unsupported, misses)


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
