"""The two LLM steps on top of the search: rerank the candidates, answer with a citation or refuse."""
from __future__ import annotations

from dataclasses import dataclass

from .llm import EFFORT, MODEL
from .loader import Article

POOL = 20     # dense candidates handed to the rerank
TOP = 5       # what the answer sees, and what Recall@5 measures

RERANK_SYSTEM = (
    "You rank articles of Swiss federal law by how directly each one answers a person's question. "
    "The question is in everyday German; the articles are the official German text. Put first the "
    "article whose text itself answers the question; then articles that answer part of it; then the "
    "rest. Judge by what the text says, not by how many words it shares with the question. Return "
    "every candidate id exactly once, as written."
)
RERANK_SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "string"}}},
    "required": ["ranking"],
    "additionalProperties": False,
}

ANSWER_SYSTEM = (
    "You answer a person's question about Swiss law using ONLY the articles given. Answer in plain "
    "German, in two to four sentences, and cite each article you rely on by its id exactly as given "
    "(e.g. \"OR Art. 335c\"). If the articles do not answer the question, set answerable to false, "
    "cite nothing, and say in one sentence that the given articles do not cover it -- do not answer "
    "from general knowledge. This is information about the law, not legal advice."
)
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answerable", "answer", "citations"],
    "additionalProperties": False,
}


def _articles_block(articles: list[Article]) -> str:
    return "\n\n".join(f"<article id=\"{a.id}\">\n{a.passage()}\n</article>" for a in articles)


def _request(task: str, system: str, schema: dict, user: str) -> dict:
    return {"task": task, "model": MODEL, "effort": EFFORT, "system": system, "schema": schema, "user": user}


@dataclass(frozen=True)
class Answer:
    answerable: bool
    text: str
    citations: list[str]     # only ids that were in the context
    unsupported: list[str]   # cited ids that were NOT in the context (dropped from citations)


class Pipeline:
    name = "dense + rerank"

    def __init__(self, dense, recorder, pool: int = POOL):
        self.dense = dense
        self.recorder = recorder
        self.pool = pool

    def candidates(self, question: str) -> list[Article]:
        return [a for a, _ in self.dense.search(question, k=self.pool)]

    def rerank(self, question: str, candidates: list[Article]) -> list[Article]:
        user = f"Question: {question}\n\nCandidates:\n\n{_articles_block(candidates)}"
        ranking = self.recorder.ask(_request("rerank", RERANK_SYSTEM, RERANK_SCHEMA, user))["ranking"]
        by_id = {a.id: a for a in candidates}
        ordered = [by_id[i] for i in dict.fromkeys(ranking) if i in by_id]
        # an id the model dropped keeps its dense position after the ranked ones
        return ordered + [a for a in candidates if a not in ordered]

    def search(self, question: str, k: int = TOP) -> list[tuple[Article, float]]:
        """The Retriever interface, so `evaluate()` scores the reranked list like any other."""
        ranked = self.rerank(question, self.candidates(question))
        return [(a, float(len(ranked) - i)) for i, a in enumerate(ranked[:k])]

    def answer(self, question: str, context: list[Article]) -> Answer:
        user = f"Question: {question}\n\nArticles:\n\n{_articles_block(context)}"
        out = self.recorder.ask(_request("answer", ANSWER_SYSTEM, ANSWER_SCHEMA, user))
        given = {a.id for a in context}
        cited = list(dict.fromkeys(out["citations"]))
        return Answer(bool(out["answerable"]), out["answer"], [c for c in cited if c in given],
                      [c for c in cited if c not in given])

    def ask(self, question: str) -> tuple[list[Article], Answer]:
        context = [a for a, _ in self.search(question, k=TOP)]
        return context, self.answer(question, context)
