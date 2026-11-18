"""A small vector store on top of numpy.

I started with ChromaDB but for a few dozen PDFs (a few thousand chunks)
a plain matrix + cosine similarity is fast enough, has zero setup, and is
easy to debug. If this grows to lakhs of chunks, swap this class for
Chroma / Qdrant - the rest of the code only uses add / search / save / load.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    def __len__(self):
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def remove_doc(self, doc_id: str):
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if self.vectors is not None and keep else None

    def doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self.chunks})

    def get_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        return [c for c in self.chunks if doc_id is None or c.doc_id == doc_id]

    def search(self, query_vector: np.ndarray, k: int = 5, doc_id: str | None = None) -> list[tuple[Chunk, float]]:
        if self.vectors is None or not self.chunks:
            return []
        scores = self.vectors @ query_vector  # vectors are normalised, so this is cosine
        order = np.argsort(-scores)
        results = []
        for i in order:
            chunk = self.chunks[i]
            if doc_id and chunk.doc_id != doc_id:
                continue
            results.append((chunk, float(scores[i])))
            if len(results) == k:
                break
        return results

    def save(self, folder: Path):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.vectors is not None:
            np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, folder: Path) -> "VectorStore":
        folder = Path(folder)
        store = cls()
        chunks_file = folder / "chunks.json"
        if not chunks_file.exists():
            return store
        store.chunks = [Chunk(**c) for c in json.loads(chunks_file.read_text(encoding="utf-8"))]
        vec_file = folder / "vectors.npy"
        store.vectors = np.load(vec_file) if vec_file.exists() else None
        return store
