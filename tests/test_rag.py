import math

from fedlex_rag.evaluate import Question, evaluate_answers, pool_recall
from fedlex_rag.loader import Article
from fedlex_rag.rag import Answer, Pipeline


def art(i):
    return Article(f"OR Art. {i}", "OR", str(i), "h", "c", f"text {i}", "u")


ARTS = [art(i) for i in range(1, 7)]


class FixedDense:
    def search(self, q, k=5):
        return [(a, 1.0) for a in ARTS[:k]]


class ScriptedRecorder:
    """Answers by task; records what it was asked."""

    def __init__(self, ranking, answer):
        self.ranking, self.answer_out, self.asked = ranking, answer, []

    def ask(self, request):
        self.asked.append(request)
        return {"ranking": self.ranking} if request["task"] == "rerank" else self.answer_out


def test_rerank_orders_dedupes_and_keeps_dropped_ids_last():
    rec = ScriptedRecorder(["OR Art. 3", "OR Art. 3", "OR Art. 99", "OR Art. 1"], {})
    p = Pipeline(FixedDense(), rec, pool=4)
    ranked = p.rerank("q", p.candidates("q"))
    assert [a.id for a in ranked] == ["OR Art. 3", "OR Art. 1", "OR Art. 2", "OR Art. 4"]
    assert '<article id="OR Art. 4">' in rec.asked[0]["user"] and rec.asked[0]["model"]


def test_answer_drops_citations_outside_the_context():
    rec = ScriptedRecorder(["OR Art. 2"], {"answerable": True, "answer": "Ja.",
                                           "citations": ["OR Art. 2", "OR Art. 2", "KVG Art. 9"]})
    context, answer = Pipeline(FixedDense(), rec, pool=6).ask("q")
    assert context[0].id == "OR Art. 2" and len(context) == 5
    assert answer == Answer(True, "Ja.", ["OR Art. 2"], ["KVG Art. 9"])


class FakePipeline:
    def __init__(self, answers):
        self.answers = answers

    def candidates(self, q):
        return [art(1), art(2)]

    def ask(self, q):
        return [], self.answers[q]


def test_answer_metrics():
    qs = [Question("q1", "a", ["OR Art. 1"]), Question("q2", "b", ["OR Art. 5"]),
          Question("q3", "c", []), Question("q4", "d", [])]
    p = FakePipeline({"a": Answer(True, "", ["OR Art. 1"], ["X"]), "b": Answer(False, "", [], []),
                      "c": Answer(False, "", [], []), "d": Answer(True, "", ["OR Art. 2"], [])})
    s = evaluate_answers(p, qs)
    assert (s.n_answerable, s.n_unanswerable) == (2, 2)
    assert s.cited == 0.5 and s.false_refusal == 0.5 and s.refusal == 0.5
    assert s.unsupported == 1 and s.misses == ["q2", "q4"]
    assert pool_recall(p, qs) == 0.5


def test_refusal_rate_without_unanswerable_questions_is_nan():
    s = evaluate_answers(FakePipeline({"a": Answer(True, "", ["OR Art. 1"], [])}),
                         [Question("q1", "a", ["OR Art. 1"])])
    assert math.isnan(s.refusal)
