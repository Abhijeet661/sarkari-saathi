"""Glue code: one object the API, UI and scripts all use."""

import hashlib
import json
import logging
from pathlib import Path

from .chunker import chunk_pages
from .config import settings
from .eligibility import check_eligibility
from .embeddings import Embedder
from .extractor import extract_details
from .llm import LLM
from .pdf_loader import Page, load_pdf
from .rag import Answer, RAGAnswerer
from .retriever import HybridRetriever
from .schemas import NotificationDetails, PostEligibility, UserProfile
from .vector_store import VectorStore

log = logging.getLogger(__name__)


def make_doc_id(path: Path) -> str:
    # name + short hash of the content, so re-uploading the same file is a no-op
    digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:8]
    stem = "".join(ch if ch.isalnum() else "-" for ch in Path(path).stem.lower()).strip("-")[:40]
    return f"{stem}-{digest}"


class Saathi:
    def __init__(self, embedder: Embedder, llm: LLM | None = None, index_dir: Path | None = None):
        self.embedder = embedder
        self.llm = llm
        self.index_dir = Path(index_dir or settings.index_dir)
        self.store = VectorStore.load(self.index_dir)
        self.retriever = HybridRetriever(self.store, embedder)
        self._details_cache: dict[str, NotificationDetails] = {}

    @classmethod
    def from_settings(cls) -> "Saathi":
        from .embeddings import make_embedder
        from .llm import GeminiLLM

        return cls(
            embedder=make_embedder(settings.embedding_provider, settings.embedding_model, settings.gemini_api_key),
            llm=GeminiLLM(settings.gemini_api_key, settings.gemini_model),
        )

    # ---- ingestion -------------------------------------------------------
    def ingest(self, pdf_path: str | Path) -> str:
        pdf_path = Path(pdf_path)
        doc_id = make_doc_id(pdf_path)
        if doc_id in self.store.doc_ids():
            log.info("%s already indexed", doc_id)
            return doc_id

        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages, doc_id, settings.chunk_size, settings.chunk_overlap)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        self._save_pages(doc_id, pages)
        self.store.save(self.index_dir)
        log.info("indexed %s: %d pages, %d chunks", doc_id, len(pages), len(chunks))
        return doc_id

    def documents(self) -> list[str]:
        return self.store.doc_ids()

    # ---- questions -------------------------------------------------------
    def ask(self, question: str, doc_id: str | None = None) -> Answer:
        self._need_llm()
        return RAGAnswerer(self.retriever, self.llm, settings.top_k).ask(question, doc_id)

    # ---- structured details + eligibility -------------------------------
    def details(self, doc_id: str) -> NotificationDetails:
        if doc_id in self._details_cache:
            return self._details_cache[doc_id]
        cache_file = self.index_dir / "details" / f"{doc_id}.json"
        if cache_file.exists():
            details = NotificationDetails.model_validate_json(cache_file.read_text(encoding="utf-8"))
        else:
            self._need_llm()
            details = extract_details(self._load_pages(doc_id), self.llm)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(details.model_dump_json(indent=1), encoding="utf-8")
        self._details_cache[doc_id] = details
        return details

    def eligibility(self, doc_id: str, profile: UserProfile) -> list[PostEligibility]:
        return check_eligibility(self.details(doc_id), profile)

    # ---- helpers ---------------------------------------------------------
    def _need_llm(self):
        if self.llm is None:
            raise RuntimeError("This needs an LLM. Set GEMINI_API_KEY in .env")

    def _save_pages(self, doc_id: str, pages: list[Page]):
        folder = self.index_dir / "pages"
        folder.mkdir(parents=True, exist_ok=True)
        data = [{"number": p.number, "text": p.text} for p in pages]
        (folder / f"{doc_id}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _load_pages(self, doc_id: str) -> list[Page]:
        data = json.loads((self.index_dir / "pages" / f"{doc_id}.json").read_text(encoding="utf-8"))
        return [Page(**p) for p in data]
