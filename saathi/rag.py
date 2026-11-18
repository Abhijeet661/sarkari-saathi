import re
from dataclasses import dataclass, field

from .llm import LLM
from .prompts import ANSWER_SYSTEM, ANSWER_TEMPLATE, NOT_FOUND_MESSAGES
from .retriever import HybridRetriever, RetrievedChunk

CITATION = re.compile(r"\[p\.?\s*(\d+)\]", re.IGNORECASE)
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# words that are common in Marathi but not in Hindi
MARATHI_HINTS = ("आहे", "काय", "किती", "मध्ये", "साठी", "आणि", "नाही", "कधी", "कोणती", "करावा")


def detect_language(text: str) -> str:
    if not DEVANAGARI.search(text):
        return "en"
    return "mr" if any(w in text for w in MARATHI_HINTS) else "hi"


@dataclass
class Answer:
    question: str
    text: str
    found: bool
    cited_pages: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list)


def build_context(hits: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[p.{h.chunk.page}]\n{h.chunk.text}" for h in hits)


def check_citations(answer_text: str, allowed_pages: set[int]) -> tuple[list[int], list[int]]:
    cited = sorted({int(p) for p in CITATION.findall(answer_text)})
    valid = [p for p in cited if p in allowed_pages]
    invalid = [p for p in cited if p not in allowed_pages]
    return valid, invalid


class RAGAnswerer:
    def __init__(self, retriever: HybridRetriever, llm: LLM, top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def ask(self, question: str, doc_id: str | None = None) -> Answer:
        question = question.strip()
        lang = detect_language(question)
        hits = self.retriever.retrieve(question, k=self.top_k, doc_id=doc_id)
        if not hits:
            return Answer(question, NOT_FOUND_MESSAGES[lang], found=False)

        prompt = ANSWER_TEMPLATE.format(context=build_context(hits), question=question)
        raw = self.llm.generate(prompt, system=ANSWER_SYSTEM)

        if raw.strip().upper().startswith("NOT_FOUND"):
            return Answer(question, NOT_FOUND_MESSAGES[lang], found=False, sources=hits)

        allowed = {h.chunk.page for h in hits}
        valid, invalid = check_citations(raw, allowed)
        # A cited page we never showed the model = it made that up. Strip it
        # so the user isn't sent to the wrong page, but keep a record.
        text = raw
        for p in invalid:
            text = re.sub(rf"\s*\[p\.?\s*{p}\]", "", text)
        return Answer(question, text.strip(), found=True, cited_pages=valid, invalid_citations=invalid, sources=hits)
