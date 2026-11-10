"""fedlex-rag: fetch the law, search it, measure the search."""
from __future__ import annotations

import argparse
import sys

from .evaluate import evaluate, load_questions
from .index import BM25, CachedEmbedder, Dense, E5Embedder
from .loader import fetch_all, load_articles, write_fixture


def _dense(articles):
    embedder = CachedEmbedder(E5Embedder())
    return Dense(articles, embedder), embedder


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


def cmd_eval(args) -> int:
    articles, questions = load_articles(), load_questions()
    dense, embedder = _dense(articles)
    rows = [evaluate(BM25(articles), questions, args.k), evaluate(dense, questions, args.k)]
    embedder.save()
    print(f"{len(questions)} questions, {rows[0].n} answerable, {len(articles)} articles\n")
    print(f"{'retriever':18} {'Recall@' + str(args.k):>9} {'hit@1':>7} {'MRR@10':>7}  misses")
    for r in rows:
        print(f"{r.name:18} {r.recall:9.1%} {r.hit1:7.1%} {r.mrr:7.3f}  {' '.join(r.misses) or '-'}")
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
    e = sub.add_parser("eval", help="measure retrieval on data/questions.jsonl")
    e.add_argument("-k", type=int, default=5)
    e.set_defaults(fn=cmd_eval)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
