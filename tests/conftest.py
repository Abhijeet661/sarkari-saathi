from pathlib import Path

import pytest

from saathi.embeddings import HashEmbedder
from saathi.pdf_loader import load_pdf

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_notification.pdf"


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    if not SAMPLE_PDF.exists():
        from scripts.make_sample_pdf import build

        build()
    return SAMPLE_PDF


@pytest.fixture(scope="session")
def sample_pages(sample_pdf):
    return load_pdf(sample_pdf)


@pytest.fixture
def embedder():
    return HashEmbedder()
