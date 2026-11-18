"""Split pages into overlapping chunks.

Chunks never cross a page boundary. That costs a little context but it means
every answer can point to one exact page number, which matters more here:
people want to open the PDF and check the age limit themselves.
"""

from dataclasses import dataclass, field

from .pdf_loader import Page


@dataclass
class Chunk:
    id: str
    doc_id: str
    page: int
    text: str
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]

    paragraphs = [p for p in text.split("\n") if p.strip()]
    pieces, current = [], ""
    for para in paragraphs:
        # a single huge paragraph (tables often come out like this) gets hard-split
        while len(para) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(para[:size])
            para = para[size - overlap:]
        if len(current) + len(para) + 1 > size and current:
            pieces.append(current)
            current = current[-overlap:] if overlap else ""
        current = f"{current}\n{para}" if current else para
    if current.strip():
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def chunk_pages(pages: list[Page], doc_id: str, size: int = 900, overlap: int = 150) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks = []
    for page in pages:
        for i, piece in enumerate(_split_text(page.text, size, overlap)):
            chunks.append(Chunk(id=f"{doc_id}:p{page.number}:{i}", doc_id=doc_id, page=page.number, text=piece))
    return chunks
