# Study plan: learn this project in 6 milestones

Do one milestone at a time. For each one: read the files, run them, change something and watch what breaks, then answer the interview questions **out loud without looking**. Move on only when you can.

Target: about 1 week per milestone.

---

## Milestone 1: Reading PDFs and chunking
**Files:** `saathi/pdf_loader.py`, `saathi/chunker.py`, `scripts/make_sample_pdf.py`, `tests/test_ingestion.py`

**Learn:** what a text layer in a PDF is, why scanned PDFs have none, what chunking is and why overlap helps.

**Try:**
- Run `python scripts/make_sample_pdf.py`, then print the pages from `load_pdf`.
- Change `size=300` to `size=100` in a test and see how many chunks you get.

**Interview questions:**
1. Why do your chunks never cross a page boundary? *(So every answer can cite one exact page. Users verify age limits themselves.)*
2. Why remove lines that repeat on most pages? *(Headers and footers add the same noise to every chunk and hurt search.)*
3. What happens with a scanned PDF? *(No text, so we raise an error. OCR is on the roadmap.)*

---

## Milestone 2: Embeddings and vector search
**Files:** `saathi/embeddings.py`, `saathi/vector_store.py`

**Learn:** what an embedding is, cosine similarity, why we normalise vectors, why a *multilingual* model.

**Try:** embed "age limit" and "आयु सीमा" with the real model and print their cosine similarity.

**Interview questions:**
1. Why not ChromaDB / Pinecone? *(A few thousand chunks fit in a numpy matrix. Zero setup, easy to debug. The class can be swapped later because the rest of the code only calls add/search/save/load.)*
2. Why `paraphrase-multilingual-MiniLM`? *(Users ask in Hindi and Marathi while the PDF is in English. A multilingual model puts both in the same vector space.)*
3. What is `HashEmbedder` for? *(Tests and CI. No model download, deterministic.)*

---

## Milestone 3: Hybrid retrieval
**Files:** `saathi/bm25.py`, `saathi/retriever.py`, `tests/test_retrieval.py`

**Learn:** TF-IDF, then BM25 (k1, b), then Reciprocal Rank Fusion.

**Try:** run `python -m eval.run_eval --fake-embedder`. Remove BM25 from the retriever and run it again. Compare MRR.

**Interview questions:**
1. Why hybrid? *(Embeddings blur exact tokens like "NIRT/REC/07/2026" or "Level 6". BM25 catches those, and vectors catch meaning.)*
2. Why RRF instead of adding scores? *(Cosine is 0–1 and BM25 can be 0–20+. RRF only uses ranks, so there's no scale problem.)*
3. What is k=60 in RRF? *(A smoothing constant from the original paper, so rank 1 doesn't dominate everything.)*

---

## Milestone 4: Grounded answers with citations
**Files:** `saathi/prompts.py`, `saathi/llm.py`, `saathi/rag.py`, `tests/test_rag.py`

**Learn:** system prompts, grounding, hallucination, temperature.

**Try:** get a free Gemini key, put it in `.env`, run the Streamlit app, and ask 10 questions (including some in Hindi and Marathi).

**Interview questions:**
1. How do you reduce hallucination? *(Answer only from excerpts, an explicit NOT_FOUND instruction, low temperature, and a check that every cited page was actually retrieved.)*
2. What happens if the model cites page 42? *(It was never in the context, so the citation is stripped and logged in `invalid_citations`.)*
3. Why `FakeLLM` in tests? *(Tests must be fast, free, deterministic and not need an API key.)*

---

## Milestone 5: Structured extraction and eligibility
**Files:** `saathi/schemas.py`, `saathi/extractor.py`, `saathi/eligibility.py`, `tests/test_extraction_and_eligibility.py`

**Learn:** Pydantic models, JSON mode, regex.

**Try:** add your own profile in a test (DOB 26 Jun 2003, B.Tech CSE) and check the results.

**Interview questions:**
1. Why isn't eligibility decided by the LLM? *(Age maths has one right answer. The LLM reads, code decides, and code can be unit-tested.)*
2. What if the LLM returns bad JSON? *(Send the Pydantic error back and let it fix its output, up to 2 retries, then raise.)*
3. What is `check_manually`? *(When we can't confidently match a qualification, we say so instead of guessing.)*

---

## Milestone 6: API, UI, evaluation, CI, deploy
**Files:** `api/main.py`, `app/streamlit_app.py`, `eval/`, `.github/workflows/ci.yml`

**Learn:** FastAPI routes and dependency injection, Streamlit caching, GitHub Actions.

**Try:**
- Run `python -m eval.run_eval --with-llm` and **put your real numbers in the README** (replace the offline numbers).
- Deploy the Streamlit app on Hugging Face Spaces or Streamlit Community Cloud, and add the live link to the README and your resume.

**Interview questions:**
1. How do you know your system works? *(The eval set measures hit@k, MRR, answer correctness, citation validity and refusal rate.)*
2. What would you improve next? *(OCR, table parsing, a cross-encoder reranker, a bigger eval set from real notifications.)*
3. What was the hardest bug? *(Pick a real one you hit while running it, for example tables coming out one cell per line, or the q14 retrieval miss.)*

---

## The 60-second project pitch (practise this)

> "While applying for government jobs I found every notification is a 40-page PDF, and the details I need are scattered. So I built Sarkari Saathi, a RAG system that answers questions about a notification with page citations, in English, Hindi or Marathi. It uses hybrid retrieval, vector search plus BM25 merged with Reciprocal Rank Fusion, because pure embeddings missed exact things like advertisement numbers. The LLM only answers from the retrieved text, and I verify that every cited page was actually retrieved. I also extract the details into validated JSON, and the eligibility check is rule-based code, not the LLM, because age maths has to be exact. Because I'd worked in LLM evaluation, I built an eval set that includes unanswerable questions, to measure whether the system refuses instead of hallucinating."
