"""Render docs/results.png from the evaluation itself -- no number in this file is typed in.

    pip install matplotlib && python docs/make_results_chart.py
Keyless: it replays the recorded answers and the committed vectors, like `make eval`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fedlex_rag.evaluate import evaluate, load_questions, pool_recall  # noqa: E402
from fedlex_rag.index import BM25, CachedEmbedder, Dense, E5Embedder  # noqa: E402
from fedlex_rag.llm import Recorder  # noqa: E402
from fedlex_rag.loader import load_articles  # noqa: E402
from fedlex_rag.rag import Pipeline  # noqa: E402

BG, INK, MUTED, GRID = "#12151b", "#e6e9ef", "#9aa3b2", "#2a303b"
CITED, NEIGHBOUR, REFUSED = "#3987e5", "#d95926", "#199e70"   # validated: dark, surface #12151b
NARROW_POOL = 20   # the pool the first recordings used, kept as a reference line

articles, questions = load_articles(), load_questions()
dense = Dense(articles, CachedEmbedder(E5Embedder()))
pipeline = Pipeline(dense, Recorder(client=None))
stages = [("BM25", evaluate(BM25(articles), questions).recall),
          ("dense (E5)", evaluate(dense, questions).recall),
          ("dense + rerank", evaluate(pipeline, questions).recall)]
ceiling = pool_recall(pipeline, questions)
narrow = pool_recall(Pipeline(dense, None, pool=NARROW_POOL), questions)

cited = neighbour = refused = unanswerable = refused_out = 0
for q in questions:
    _, a = pipeline.ask(q.question)
    if q.expected:
        if a.answerable and set(a.citations) & set(q.expected):
            cited += 1
        elif a.answerable:
            neighbour += 1
        else:
            refused += 1
    else:
        unanswerable += 1
        refused_out += not a.answerable
answerable = cited + neighbour + refused


def pct(v: float) -> str:
    return f"{v * 100:.1f} %"


def bar(ax, x, y, w, h, colour):
    """A thin bar with small rounded ends and a 2 px surface gap around it (stacked segments too)."""
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=colour, edgecolor=BG, linewidth=2, mutation_aspect=8))


fig = plt.figure(figsize=(12.8, 5.6), dpi=100, facecolor=BG)
fig.text(0.035, 0.93, "fedlex-rag on 76 questions about Swiss federal law", color=INK, fontsize=17,
         fontweight="bold")
fig.text(0.035, 0.875, f"{len(articles)} articles of the DSG, the OR's employment contract and the KVG  ·  "
         "rerank and answer by claude-opus-5-5, replayed from recordings", color=MUTED, fontsize=11)

# left: Recall@5 per stage, with the two ceilings
ax = fig.add_axes([0.15, 0.14, 0.34, 0.6], facecolor=BG)
for x in (0.25, 0.5, 0.75, 1.0):
    ax.axvline(x, color=GRID, linewidth=0.8, zorder=0, ymax=0.9)
    ax.text(x, -0.62, f"{x * 100:.0f} %", color=MUTED, fontsize=9, ha="center", va="top")
for i, (name, value) in enumerate(reversed(stages)):
    bar(ax, 0, i, value, 0.46, CITED)
    ax.text(-0.02, i, name, color=INK, fontsize=12, ha="right", va="center")
    ax.text(value - 0.015, i, pct(value), color=BG, fontsize=12, ha="right", va="center", fontweight="bold")
for x in (narrow, ceiling):
    ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), ymax=0.9)
ax.text(narrow - 0.01, 2.62, f"{NARROW_POOL} candidates\n{pct(narrow)}", color=MUTED, fontsize=9.5,
        ha="right", va="bottom")
ax.text(ceiling + 0.01, 2.62, f"rerank ceiling\n{pipeline.pool} candidates: {pct(ceiling)}", color=MUTED,
        fontsize=9.5, ha="left", va="bottom")
ax.set_xlim(0, 1)
ax.set_ylim(-0.6, 3.15)
ax.set_title("Recall@5: the right article in the top five", color=INK, fontsize=12.5, loc="left", pad=10)
ax.axis("off")

# right: what the answer did
ax2 = fig.add_axes([0.62, 0.14, 0.35, 0.6], facecolor=BG)
rows = [(f"{answerable} answerable", [(cited, CITED), (neighbour, NEIGHBOUR), (refused, REFUSED)], answerable),
        (f"{unanswerable} not covered", [(refused_out, REFUSED), (unanswerable - refused_out, NEIGHBOUR)],
         unanswerable)]
for i, (label, parts, total) in enumerate(rows):
    y = 1 - i
    x = 0.0
    for n, colour in parts:
        if not n:
            continue
        w = n / total
        bar(ax2, x, y, w, 0.42, colour)
        if w > 0.08:
            ax2.text(x + w / 2, y, str(n), color=BG, fontsize=11.5, ha="center", va="center", fontweight="bold")
        x += w
    ax2.text(0, y + 0.34, label, color=INK, fontsize=12, va="bottom")
ax2.text(cited / answerable, 1 - 0.34, f"{pct(cited / answerable)} cite a right article", color=MUTED,
         fontsize=9.5, ha="right", va="top")
ax2.text(1, -0.34, f"{refused_out} of {unanswerable} refused", color=MUTED, fontsize=9.5, ha="right", va="top")
ax2.set_xlim(0, 1)
ax2.set_ylim(-1.05, 1.75)
ax2.set_title("The answer: cite, or refuse", color=INK, fontsize=12.5, loc="left", pad=10)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (CITED, NEIGHBOUR, REFUSED)]
ax2.legend(handles, ["right article", "neighbour article / answers", "refuses"], loc="lower left",
           bbox_to_anchor=(0, 0), ncol=3, frameon=False, labelcolor=INK, fontsize=9.5, handlelength=1.2)
ax2.axis("off")

out = Path(__file__).with_name("results.png")
fig.savefig(out, facecolor=BG)
print(out)
