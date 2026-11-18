"""Read a notification PDF into a list of pages.

Most recruitment PDFs repeat the same header/footer on every page
("Advt No. 10/2026", "Page 3 of 12", website links). Those lines add noise
to every chunk, so we drop lines that show up on most pages.
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    number: int  # 1-based, same as what a person sees in the PDF viewer
    text: str


PAGE_NO_PATTERN = re.compile(r"^\s*(page\s*)?\d+\s*(of|/)\s*\d+\s*$", re.IGNORECASE)


def _clean_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line and not PAGE_NO_PATTERN.match(line):
            lines.append(line)
    return lines


def remove_repeated_lines(pages_lines: list[list[str]], threshold: float = 0.6) -> list[list[str]]:
    if len(pages_lines) < 3:
        return pages_lines
    counts = Counter()
    for lines in pages_lines:
        counts.update(set(lines))
    limit = threshold * len(pages_lines)
    repeated = {line for line, c in counts.items() if c >= limit}
    return [[l for l in lines if l not in repeated] for lines in pages_lines]


def load_pdf(path: str | Path) -> list[Page]:
    reader = PdfReader(str(path))
    raw = [_clean_lines(p.extract_text() or "") for p in reader.pages]
    cleaned = remove_repeated_lines(raw)
    pages = [Page(number=i + 1, text="\n".join(lines)) for i, lines in enumerate(cleaned)]

    if not any(p.text for p in pages):
        # scanned PDF - no text layer. OCR is on the todo list.
        raise ValueError(f"No text found in {path}. It is probably a scanned PDF.")
    return pages
