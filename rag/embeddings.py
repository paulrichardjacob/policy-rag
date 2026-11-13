"""Two interchangeable embedders.

MiniLMEmbedder  - sentence-transformers model, real semantic embeddings.
TfidfEmbedder   - scikit-learn TF-IDF. No model download, so the project
                  (and the tests) run on any machine, including CI.

Both return L2-normalised vectors, so a dot product is cosine similarity.
"""

import pickle
from pathlib import Path

import numpy as np


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.fitted = False

    def fit(self, texts: list):
        self.vectorizer.fit(texts)
        self.fitted = True
        return self

    def encode(self, texts: list) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TfidfEmbedder must be fitted on the corpus before encoding")
        # TfidfVectorizer already L2-normalises each row.
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def save(self, folder: Path):
        with open(Path(folder) / "tfidf.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, folder: Path, model_name=None):
        obj = cls()
        # Only ever load an index you built yourself: pickle can run code.
        with open(Path(folder) / "tfidf.pkl", "rb") as f:
            obj.vectorizer = pickle.load(f)
        obj.fitted = True
        return obj


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list):
        return self  # pretrained, nothing to fit

    def encode(self, texts: list) -> np.ndarray:
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def save(self, folder: Path):
        pass  # the model is cached by sentence-transformers

    @classmethod
    def load(cls, folder: Path, model_name=None):
        return cls(model_name or "sentence-transformers/all-MiniLM-L6-v2")


EMBEDDERS = {"tfidf": TfidfEmbedder, "minilm": MiniLMEmbedder}


def minilm_installed() -> bool:
    try:
        import sentence_transformers  # noqa: F401

        return True
    except ImportError:
        return False


def make_embedder(kind: str = "auto", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if kind == "auto":
        kind = "minilm" if minilm_installed() else "tfidf"
    if kind == "minilm":
        return MiniLMEmbedder(model_name)
    if kind == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedder '{kind}'. Use auto, minilm or tfidf.")
