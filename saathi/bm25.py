"""Okapi BM25 keyword search.

Vector search alone kept missing exact things like "Advt No. 10/2026",
"Level 6" or "EWS" - short tokens that embeddings blur together. BM25 is
good at exactly those, so retrieval uses both (see retriever.py).
"""

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in documents]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / len(self.docs) if self.docs else 0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.doc_len):
            s = 0.0
            for t in terms:
                if t not in tf:
                    continue
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf[t] * freq * (self.k1 + 1) / denom
            out.append(s)
        return out
