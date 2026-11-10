"""Two retrievers over the articles: BM25 (the keyword baseline) and dense embeddings.

The dense embedder is a local model (multilingual E5), so search needs no API key. Its vectors
are cached in `data/embeddings/`; a cache that covers the corpus and the evaluation questions lets
`make eval` run without the model installed at all.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .loader import DATA, Article

E5_SMALL = "intfloat/multilingual-e5-small"
CACHE_DIR = DATA / "embeddings"

_STOP_WORDS = """der die das den dem des ein eine einer einem einen eines und oder aber auch als am an auf aus bei
bis da dass du er es für hat habe ich ihr im in ist ja kann man mich mir mit muss nach nicht noch nur ob sie so um
von vor was wenn wer wie wir wird zu zum zur über sich sind sein seine werden welche welcher welches mein meine
meinen meiner"""
_STOP = frozenset(_STOP_WORDS.split())


def tokens(text: str) -> list[str]:
    """Lower-case words without stop words, with a crude German suffix strip (Ferien/Ferie -> feri)."""
    out = []
    for w in re.findall(r"[a-zäöüß0-9]+", text.lower()):
        if w in _STOP or len(w) < 2:
            continue
        for suffix in ("ungen", "ung", "en", "er", "es", "e", "n", "s"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.append(w)
    return out


class BM25:
    name = "bm25"

    def __init__(self, articles: list[Article], k1: float = 1.5, b: float = 0.75):
        self.articles = articles
        self.docs = [Counter(tokens(a.passage())) for a in articles]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 5) -> list[tuple[Article, float]]:
        q = tokens(query)
        scores = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in q:
                f = doc.get(t, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * length / self.avg)
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            scores.append(s)
        order = np.argsort(scores)[::-1][:k]
        return [(self.articles[i], float(scores[i])) for i in order]


class E5Embedder:
    """multilingual E5: queries and passages carry their role as a prefix; vectors are unit length."""

    def __init__(self, model: str = E5_SMALL):
        self.model_name = model
        self._model = None

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - depends on the install
                raise RuntimeError("the embedding model is not installed: pip install -e '.[embed]' "
                                   "(or use the committed cache, which covers the evaluation set)") from exc
            self._model = SentenceTransformer(self.model_name)
        prefixed = [f"{kind}: {t}" for t in texts]
        return np.asarray(self._model.encode(prefixed, normalize_embeddings=True, batch_size=32), dtype=np.float32)


class CachedEmbedder:
    """Looks each (kind, text) up in an .npz cache first; only the misses reach the model."""

    def __init__(self, base, path: Path | None = None):
        self.base = base
        name = getattr(base, "model_name", "embedder").split("/")[-1]
        self.path = path or CACHE_DIR / f"{name}.npz"
        self.cache: dict[str, np.ndarray] = {}
        self.dirty = False
        if self.path.exists():
            with np.load(self.path) as z:
                self.cache = dict(zip(z["keys"].tolist(), z["vectors"].astype(np.float32), strict=True))

    @staticmethod
    def key(text: str, kind: str) -> str:
        return hashlib.sha1(f"{kind}\0{text}".encode()).hexdigest()

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        keys = [self.key(t, kind) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        if missing:
            vectors = self.base.encode([texts[i] for i in missing], kind)
            for i, v in zip(missing, vectors, strict=True):
                self.cache[keys[i]] = v
            self.dirty = True
        return np.stack([self.cache[k] for k in keys])

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.cache)
        np.savez_compressed(self.path, keys=np.array(keys), vectors=np.stack([self.cache[k] for k in keys])
                            .astype(np.float16))
        self.dirty = False


class Dense:
    name = "dense"

    def __init__(self, articles: list[Article], embedder):
        self.articles = articles
        self.embedder = embedder
        self.matrix = embedder.encode([a.passage() for a in articles], "passage")

    def search(self, query: str, k: int = 5) -> list[tuple[Article, float]]:
        q = self.embedder.encode([query], "query")[0]
        scores = self.matrix @ q
        order = np.argsort(scores)[::-1][:k]
        return [(self.articles[i], float(scores[i])) for i in order]
