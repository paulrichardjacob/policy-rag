"""The index: chunk embeddings on disk plus dense, BM25 and hybrid search.

The vector store is a plain NumPy matrix. For a few thousand chunks an exact
dot product is faster than setting up a vector database and it keeps the
retrieval logic easy to read. Swapping in FAISS or Chroma later only means
replacing `_dense_scores`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunker import Chunk
from .embeddings import EMBEDDERS

MODES = ("hybrid", "dense", "bm25")
RRF_K = 60  # standard constant for reciprocal rank fusion


@dataclass
class Hit:
    chunk: Chunk
    score: float  # ranking score for the chosen mode
    similarity: float  # cosine similarity between query and chunk


class VectorIndex:
    def __init__(self, chunks: list, embeddings: np.ndarray, embedder):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.bm25 = BM25([c.indexed_text for c in chunks])

    @classmethod
    def build(cls, chunks: list, embedder):
        if not chunks:
            raise ValueError("No chunks to index")
        texts = [c.indexed_text for c in chunks]
        embedder.fit(texts)
        return cls(chunks, embedder.encode(texts), embedder)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "embeddings.npy", self.embeddings)
        (folder / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], indent=1), encoding="utf-8"
        )
        meta = {
            "embedder": self.embedder.name,
            "model_name": getattr(self.embedder, "model_name", None),
            "num_chunks": len(self.chunks),
            "sources": sorted({c.source for c in self.chunks}),
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        self.embedder.save(folder)

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        if not (folder / "meta.json").exists():
            raise FileNotFoundError(f"No index in {folder}. Run: python cli.py ingest")
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
        embeddings = np.load(folder / "embeddings.npy")
        embedder = EMBEDDERS[meta["embedder"]].load(folder, meta.get("model_name"))
        return cls(chunks, embeddings, embedder)

    def _dense_scores(self, query: str) -> np.ndarray:
        return self.embeddings @ self.embedder.encode([query])[0]

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        dense = self._dense_scores(query)
        if mode == "dense":
            ranking = dense
        elif mode == "bm25":
            ranking = np.asarray(self.bm25.scores(query))
        else:
            ranking = self._rrf(dense, np.asarray(self.bm25.scores(query)))
        order = np.argsort(-ranking, kind="stable")[:k]
        return [Hit(self.chunks[i], float(ranking[i]), float(dense[i])) for i in order]

    @staticmethod
    def _rrf(dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
        """Reciprocal rank fusion: add 1 / (60 + rank) from each ranking.

        Ranks are used instead of raw scores because cosine similarity and
        BM25 are on different scales. Chunks with a BM25 score of zero get
        nothing from the keyword side.
        """
        fused = np.zeros(len(dense))
        for rank, i in enumerate(np.argsort(-dense, kind="stable")):
            fused[i] += 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(np.argsort(-sparse, kind="stable")):
            if sparse[i] > 0:
                fused[i] += 1.0 / (RRF_K + rank + 1)
        return fused
