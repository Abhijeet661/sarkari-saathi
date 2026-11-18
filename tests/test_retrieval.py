import numpy as np

from saathi.bm25 import BM25, tokenize
from saathi.chunker import chunk_pages
from saathi.retriever import HybridRetriever, reciprocal_rank_fusion
from saathi.vector_store import VectorStore


def test_tokenizer_keeps_advt_numbers():
    assert "nirt/rec/07/2026" in tokenize("Advertisement No. NIRT/REC/07/2026")


def test_bm25_prefers_matching_doc():
    bm = BM25(["application fee is 500", "age limit is 30 years", "selection by CBT"])
    scores = bm.scores("what is the age limit")
    assert scores.index(max(scores)) == 1


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    assert fused["a"] == fused["b"]  # both ranks 1 and 2
    assert fused["b"] > fused["c"] > 0
    assert fused["d"] < fused["a"]


def test_vector_store_save_and_load(tmp_path, sample_pages, embedder):
    chunks = chunk_pages(sample_pages, "doc")
    store = VectorStore()
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    store.save(tmp_path)

    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == len(store)
    assert np.allclose(loaded.vectors, store.vectors)


def test_hybrid_finds_right_page(sample_pages, embedder):
    chunks = chunk_pages(sample_pages, "doc")
    store = VectorStore()
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    retriever = HybridRetriever(store, embedder)

    hits = retriever.retrieve("application fee for general candidates", k=3)
    assert hits[0].chunk.page == 4

    hits = retriever.retrieve("age relaxation for OBC", k=3)
    assert 3 in [h.chunk.page for h in hits]


def test_doc_filter(sample_pages, embedder):
    store = VectorStore()
    for doc in ("a", "b"):
        chunks = chunk_pages(sample_pages, doc)
        store.add(chunks, embedder.embed([c.text for c in chunks]))
    hits = HybridRetriever(store, embedder).retrieve("fee", k=5, doc_id="b")
    assert hits and all(h.chunk.doc_id == "b" for h in hits)
