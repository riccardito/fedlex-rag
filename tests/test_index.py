import numpy as np

from fedlex_rag.index import BM25, CachedEmbedder, Dense, tokens
from fedlex_rag.loader import Article


def art(i, text):
    return Article(f"OR Art. {i}", "OR", str(i), "", "", text, "u")


ARTS = [art(1, "Ferien: vier Wochen pro Dienstjahr"), art(2, "Kündigung während der Probezeit"),
        art(3, "Lohn bei Krankheit des Arbeitnehmers")]


def test_tokens_strip_stop_words_and_suffixes():
    assert tokens("Die Kündigungen und der Lohn") == ["kündig", "lohn"]


def test_bm25_ranks_the_matching_article_first():
    hits = BM25(ARTS).search("Wie viele Wochen Ferien?", k=2)
    assert hits[0][0].id == "OR Art. 1" and hits[0][1] > hits[1][1]


class FakeEmbedder:
    """Deterministic 3-d vectors: one axis per keyword."""

    model_name = "org/fake-model"

    def __init__(self):
        self.calls = 0

    def encode(self, texts, kind):
        self.calls += len(texts)
        words = ("ferien", "probezeit", "krank")
        v = np.array([[float(w in t.lower()) for w in words] for t in texts], dtype=np.float32) + 0.01
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_dense_search_with_a_fake_embedder():
    hits = Dense(ARTS, FakeEmbedder()).search("bin ich krank versichert", k=1)
    assert hits[0][0].id == "OR Art. 3"


def test_cache_hits_skip_the_model_and_survive_a_save(tmp_path):
    base = FakeEmbedder()
    cached = CachedEmbedder(base, tmp_path / "c.npz")
    first = cached.encode(["Ferien", "Probezeit"], "passage")
    cached.encode(["Ferien"], "passage")
    assert base.calls == 2
    cached.save()
    reloaded = CachedEmbedder(FakeEmbedder(), tmp_path / "c.npz")
    again = reloaded.encode(["Ferien", "Probezeit"], "passage")
    assert reloaded.base.calls == 0
    assert np.allclose(first, again, atol=1e-3)   # float16 on disk
    assert reloaded.encode(["Ferien"], "query").shape == (1, 3) and reloaded.base.calls == 1


def test_cache_name_and_clean_save(tmp_path, monkeypatch):
    monkeypatch.setattr("fedlex_rag.index.CACHE_DIR", tmp_path)
    cached = CachedEmbedder(FakeEmbedder())
    assert cached.path.name == "fake-model.npz"
    cached.save()                      # nothing to write yet
    assert not cached.path.exists()
