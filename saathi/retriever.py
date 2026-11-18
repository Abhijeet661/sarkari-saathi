"""Hybrid retrieval: vector search + BM25, merged with Reciprocal Rank Fusion.

RRF only looks at the rank of each chunk in each list, not the raw scores,
so we don't have to worry that cosine scores (0-1) and BM25 scores (0-20+)
live on different scales.
"""

from dataclasses import dataclass

from .bm25 import BM25
from .chunker import Chunk
from .embeddings import Embedder
from .vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    vector_rank: int | None
    bm25_rank: int | None


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, candidates: int = 20):
        self.store = store
        self.embedder = embedder
        self.candidates = candidates

    def retrieve(self, query: str, k: int = 5, doc_id: str | None = None) -> list[RetrievedChunk]:
        pool = self.store.get_chunks(doc_id)
        if not pool:
            return []

        # some embedders (Gemini) use a different mode for queries than for documents
        embed_query = getattr(self.embedder, "embed_query", self.embedder.embed)
        qvec = embed_query([query])[0]
        vector_hits = self.store.search(qvec, k=self.candidates, doc_id=doc_id)
        vector_ids = [c.id for c, _ in vector_hits]

        # BM25 is rebuilt per query. With a few thousand chunks this takes
        # milliseconds, and it avoids keeping a second index in sync.
        bm25 = BM25([c.text for c in pool])
        scored = sorted(zip(pool, bm25.scores(query)), key=lambda x: -x[1])
        bm25_ids = [c.id for c, s in scored[: self.candidates] if s > 0]

        fused = reciprocal_rank_fusion([vector_ids, bm25_ids])
        by_id = {c.id: c for c in pool}
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [
            RetrievedChunk(
                chunk=by_id[cid],
                score=score,
                vector_rank=vector_ids.index(cid) + 1 if cid in vector_ids else None,
                bm25_rank=bm25_ids.index(cid) + 1 if cid in bm25_ids else None,
            )
            for cid, score in top
        ]
