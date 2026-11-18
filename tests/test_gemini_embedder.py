from types import SimpleNamespace

import numpy as np

from saathi.embeddings import GeminiEmbedder


class FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, model, contents, config):
        self.calls.append((len(contents), config.task_type))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0]) for _ in contents])


def make_fake(batch=100):
    emb = GeminiEmbedder.__new__(GeminiEmbedder)  # skip __init__, no API key / network
    emb.client = SimpleNamespace(models=FakeModels())
    emb._types = SimpleNamespace(EmbedContentConfig=lambda task_type, output_dimensionality: SimpleNamespace(task_type=task_type))
    emb.model, emb.dim, emb.BATCH = "fake", 2, batch
    return emb


def test_vectors_are_normalised():
    vecs = make_fake().embed(["a", "b"])
    assert vecs.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_batches_and_task_types():
    emb = make_fake(batch=2)
    emb.embed(["a", "b", "c"])
    emb.embed_query(["q"])
    assert emb.client.models.calls == [(2, "RETRIEVAL_DOCUMENT"), (1, "RETRIEVAL_DOCUMENT"), (1, "RETRIEVAL_QUERY")]
