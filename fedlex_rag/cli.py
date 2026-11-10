"""fedlex-rag: fetch the law, search it, rerank and answer with a citation, and measure each step."""
from __future__ import annotations

import argparse
import sys

from .evaluate import evaluate, evaluate_answers, load_questions, pool_recall
from .index import BM25, CachedEmbedder, Dense, E5Embedder
from .llm import EFFORT, MODEL, MissingRecording, Recorder, live_client
from .loader import fetch_all, load_articles, write_fixture
from .rag import POOL, Pipeline


def _dense(articles):
    embedder = CachedEmbedder(E5Embedder())
    return Dense(articles, embedder), embedder


def _pipeline(articles, live: bool):
    dense, embedder = _dense(articles)
    recorder = Recorder(client=live_client() if live else None)
    return Pipeline(dense, recorder), embedder, recorder


def cmd_fetch(args) -> int:
    articles, manifest = fetch_all()
    write_fixture(articles, manifest)
    for m in manifest:
        print(f"{m['abbr']:4} SR {m['sr']:7} in force since {m['applicable_from']}: {m['articles']} articles")
    return 0


def cmd_search(args) -> int:
    articles = load_articles()
    if args.bm25:
        retriever = BM25(articles)
    else:
        retriever, embedder = _dense(articles)
        embedder.save()
    for article, score in retriever.search(args.query, k=args.k):
        print(f"{score:7.3f}  {article.id:16} {article.heading}\n          {article.url}")
    return 0


def cmd_ask(args) -> int:
    pipeline, embedder, recorder = _pipeline(load_articles(), args.live)
    try:
        context, answer = pipeline.ask(args.question)
    except MissingRecording:
        print("This question has no recorded answer. Run it with --live (needs an API key).", file=sys.stderr)
        return 2
    finally:
        embedder.save()
        recorder.save()
    print(answer.text + "\n")
    for a in context:
        mark = "*" if a.id in answer.citations else " "
        print(f" {mark} {a.id:16} {a.url}")
    return 0


def cmd_eval(args) -> int:
    articles, questions = load_articles(), load_questions()
    pipeline, embedder, recorder = _pipeline(articles, args.live)
    complete = False
    try:
        rows = [evaluate(BM25(articles), questions, args.k), evaluate(pipeline.dense, questions, args.k)]
        if not args.search_only:
            rows.append(evaluate(pipeline, questions, args.k))
            answers = evaluate_answers(pipeline, questions)
            ceiling = pool_recall(pipeline, questions)
        complete = True
    except MissingRecording as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    finally:
        embedder.save()
        # prune only after a run that asked for everything: an aborted run never deletes a recording
        recorder.save(prune=args.live and complete and not args.search_only)
    print(f"{len(questions)} questions ({rows[0].n} answerable), {len(articles)} articles"
          + ("" if args.search_only else f"; rerank and answer by {MODEL}, effort {EFFORT}") + "\n")
    print(f"{'retrieval':18} {'Recall@' + str(args.k):>9} {'hit@1':>7} {'MRR@10':>7}  misses")
    for r in rows:
        print(f"{r.name:18} {r.recall:9.1%} {r.hit1:7.1%} {r.mrr:7.3f}  {' '.join(r.misses) or '-'}")
    if args.search_only:
        return 0
    print(f"\nthe rerank's ceiling: an expected article is among the {POOL} dense candidates "
          f"for {ceiling:.1%} of the answerable questions")
    print(f"\n{'answer':18} {'share':>9}")
    print(f"{'cites a right article':24} {answers.cited:6.1%}   (of {answers.n_answerable} answerable)")
    print(f"{'refused, answerable':24} {answers.false_refusal:6.1%}")
    print(f"{'refused, not covered':24} {answers.refusal:6.1%}   (of {answers.n_unanswerable} unanswerable)")
    print(f"{'unsupported citations':24} {answers.unsupported:6d}")
    print(f"misses: {' '.join(answers.misses) or '-'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fedlex-rag", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the three acts from fedlex and rewrite data/")
    f.set_defaults(fn=cmd_fetch)
    s = sub.add_parser("search", help="the k best articles for a question")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--bm25", action="store_true", help="keyword search instead of embeddings")
    s.set_defaults(fn=cmd_search)
    a = sub.add_parser("ask", help="search, rerank and answer one question")
    a.add_argument("question")
    a.add_argument("--live", action="store_true", help="call Claude for an unrecorded question")
    a.set_defaults(fn=cmd_ask)
    e = sub.add_parser("eval", help="measure every step on data/questions.jsonl")
    e.add_argument("-k", type=int, default=5)
    e.add_argument("--live", action="store_true", help="call Claude for requests with no recording")
    e.add_argument("--search-only", action="store_true", help="skip the rerank and the answer")
    e.set_defaults(fn=cmd_eval)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
