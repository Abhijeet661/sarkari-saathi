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


def test_retry_on_overload_then_success():
    from saathi.llm import with_retry

    class Busy(Exception):
        code = 503

    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise Busy()
        return "ok"

    assert with_retry(flaky, sleep=waits.append) == "ok"
    assert waits == [2.0, 4.0]


def test_retry_on_timeout():
    from saathi.llm import with_retry

    class ReadTimeout(Exception):
        pass

    calls, waits = [], []

    def slow_then_ok():
        calls.append(1)
        if len(calls) == 1:
            raise ReadTimeout()
        return "ok"

    assert with_retry(slow_then_ok, sleep=waits.append) == "ok"
    assert waits == [2.0]


def test_no_retry_on_bad_request():
    import pytest

    from saathi.llm import with_retry

    class BadKey(Exception):
        code = 400

    waits = []
    with pytest.raises(BadKey):
        with_retry(lambda: (_ for _ in ()).throw(BadKey()), sleep=waits.append)
    assert waits == []
