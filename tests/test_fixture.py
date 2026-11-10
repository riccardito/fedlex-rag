"""The committed data: the fixture, the questions, and the vector cache that keeps `make eval` keyless."""
import json
import re

from fedlex_rag.cli import main
from fedlex_rag.evaluate import load_questions
from fedlex_rag.index import CachedEmbedder, Dense, E5Embedder
from fedlex_rag.loader import MANIFEST, load_articles
from fedlex_rag.sources import BY_ABBR


def test_fixture_covers_the_three_acts_within_their_ranges():
    articles = load_articles()
    assert {a.act for a in articles} == {"DSG", "OR", "KVG"}
    assert len({a.id for a in articles}) == len(articles)
    for a in articles:
        assert BY_ABBR[a.act].keeps(int(re.match(r"\d+", a.number).group()))
        assert a.text and a.url.startswith("https://www.fedlex.admin.ch/eli/cc/")
    manifest = json.loads(MANIFEST.read_text("utf-8"))
    assert "URG" in manifest["note"]
    assert sum(s["articles"] for s in manifest["sources"]) == len(articles)


def test_every_expected_article_exists():
    ids = {a.id for a in load_articles()}
    for q in load_questions():
        assert set(q.expected) <= ids, q.id


def test_the_cache_covers_corpus_and_questions():
    """Without this, `make eval` would need the model -- and CI does not install it."""
    cache = CachedEmbedder(E5Embedder())
    keys = [cache.key(a.passage(), "passage") for a in load_articles()]
    keys += [cache.key(q.question, "query") for q in load_questions()]
    assert set(keys) <= set(cache.cache)


def test_cli_eval_runs_from_the_cache(capsys):
    assert main(["eval"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^dense\s+\d+\.\d%", out, re.M) and re.search(r"^bm25\s+\d+\.\d%", out, re.M)


def test_cli_eval_search_only(capsys):
    assert main(["eval", "--search-only"]) == 0
    assert "dense + rerank" not in capsys.readouterr().out


def test_cli_ask_replays_a_recorded_question(capsys):
    q = next(q for q in load_questions() if q.id == "q033")
    assert main(["ask", q.question]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^ \* OR Art\. 323 ", out, re.M)       # the cited article is starred


def test_cli_ask_without_a_recording_says_how(capsys):
    assert main(["ask", "Eine Frage, die nie aufgezeichnet wurde?"]) == 2
    assert "--live" in capsys.readouterr().err


def test_every_question_has_its_recordings():
    """Without this, CI's keyless `make eval` would stop at the first unrecorded question."""
    from fedlex_rag.llm import Recorder
    from fedlex_rag.rag import Pipeline

    pipeline = Pipeline(Dense(load_articles(), CachedEmbedder(E5Embedder())), Recorder(client=None))
    for q in load_questions():
        pipeline.ask(q.question)      # raises MissingRecording on a gap


def test_cli_keyword_search(capsys):
    assert main(["search", "Kündigungsfrist Probezeit", "--bm25", "-k", "3"]) == 0
    assert "OR Art. 335b" in capsys.readouterr().out
