import hashlib
import re
from typing import Protocol

import numpy as np

from .llm import with_retry


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Real embeddings. Loaded lazily because the model takes a few seconds."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)


class GeminiEmbedder:
    """Embeddings from the Gemini API.

    Used on Streamlit Community Cloud, where the free tier has about 1 GB of
    memory: torch + a local model doesn't fit there. gemini-embedding-001 is
    multilingual, so Hindi / Marathi questions still match English text.
    """

    BATCH = 100

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", dim: int = 768):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Copy .env.example to .env and add your key.")
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = dim

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        rows = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i:i + self.BATCH]
            response = with_retry(lambda: self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=self._types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
            ))
            rows.extend(e.values for e in response.embeddings)
        vectors = np.asarray(rows, dtype=np.float32)
        # vectors below the full 3072 dims come back un-normalised
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_QUERY")


def make_embedder(provider: str, model_name: str, api_key: str) -> Embedder:
    """provider: "local", "gemini" or "auto" (local if sentence-transformers is installed)."""
    if provider == "auto":
        try:
            import sentence_transformers  # noqa: F401

            provider = "local"
        except ImportError:
            provider = "gemini"
    if provider == "gemini":
        return GeminiEmbedder(api_key)
    return SentenceTransformerEmbedder(model_name)


class HashEmbedder:
    """Cheap bag-of-words embedder for tests and CI (no model download).

    Not meant for real use - it only matches exact words.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
                out[row, idx] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms
