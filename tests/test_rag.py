from saathi.llm import FakeLLM
from saathi.pipeline import Saathi
from saathi.rag import check_citations, detect_language


def make_saathi(tmp_path, embedder, sample_pdf, replies):
    saathi = Saathi(embedder, FakeLLM(replies), index_dir=tmp_path)
    doc_id = saathi.ingest(sample_pdf)
    return saathi, doc_id


def test_language_detection():
    assert detect_language("What is the fee?") == "en"
    assert detect_language("आयु सीमा क्या है?") == "hi"
    assert detect_language("शेवटची तारीख कोणती आहे?") == "mr"


def test_citation_check():
    valid, invalid = check_citations("Fee is Rs. 500 [p.4]. Apply online [p.9].", {3, 4})
    assert valid == [4] and invalid == [9]


def test_answer_with_valid_citation(tmp_path, embedder, sample_pdf):
    saathi, doc_id = make_saathi(tmp_path, embedder, sample_pdf, ["The fee is Rs. 500 [p.4]."])
    ans = saathi.ask("What is the application fee?", doc_id)
    assert ans.found
    assert ans.cited_pages == [4]
    assert "[p.4]" in ans.text


def test_invented_page_is_stripped(tmp_path, embedder, sample_pdf):
    saathi, doc_id = make_saathi(tmp_path, embedder, sample_pdf, ["The fee is Rs. 500 [p.4] [p.42]."])
    ans = saathi.ask("What is the application fee?", doc_id)
    assert ans.invalid_citations == [42]
    assert "[p.42]" not in ans.text


def test_not_found_uses_question_language(tmp_path, embedder, sample_pdf):
    saathi, doc_id = make_saathi(tmp_path, embedder, sample_pdf, ["NOT_FOUND"])
    ans = saathi.ask("परीक्षा केंद्र कहाँ है?", doc_id)
    assert not ans.found
    assert "अधिसूचना" in ans.text


def test_reingesting_same_file_is_skipped(tmp_path, embedder, sample_pdf):
    saathi, doc_id = make_saathi(tmp_path, embedder, sample_pdf, [])
    before = len(saathi.store)
    assert saathi.ingest(sample_pdf) == doc_id
    assert len(saathi.store) == before
