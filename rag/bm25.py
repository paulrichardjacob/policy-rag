"""Okapi BM25 keyword search, written from scratch.

Dense embeddings are good at meaning but can miss exact terms such as
"NEFT" or "Rs 2,000". BM25 catches those, which is why the retriever
combines the two.
"""

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it my of on or "
    "the to what when which who will with".split()
)


def tokenize(text: str) -> list:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, texts: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        n = len(self.docs)
        df = Counter(term for d in self.docs for term in d)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list:
        terms = tokenize(query)
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            out.append(score)
        return out
