from saathi.chunker import chunk_pages
from saathi.pdf_loader import Page, remove_repeated_lines


def test_sample_pdf_has_five_pages(sample_pages):
    assert len(sample_pages) == 5
    assert "NIRT/REC/07/2026" in sample_pages[0].text


def test_repeated_header_is_removed():
    pages = [["Advt No. 10/2026", f"content {i}"] for i in range(5)]
    cleaned = remove_repeated_lines(pages)
    assert all("Advt No. 10/2026" not in p for p in cleaned)
    assert cleaned[2] == ["content 2"]


def test_chunks_keep_page_numbers(sample_pages):
    chunks = chunk_pages(sample_pages, "doc", size=300, overlap=50)
    assert {c.page for c in chunks} == {1, 2, 3, 4, 5}
    assert all(len(c.text) <= 300 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_long_paragraph_is_split_with_overlap():
    page = Page(number=1, text="x" * 1000)
    chunks = chunk_pages([page], "doc", size=400, overlap=100)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 400 for c in chunks)
