"""Evaluate retrieval and answers on the sample notification.

    python -m eval.run_eval                  # retrieval only, real embeddings
    python -m eval.run_eval --with-llm       # + answer checks (needs GEMINI_API_KEY)
    python -m eval.run_eval --fake-embedder  # quick offline run (used in CI)

Metrics
- hit@k: is one of the expected pages in the top k chunks?
- MRR: 1 / rank of the first chunk from an expected page (higher = found earlier)
- answer_correct: answer contains every must_include string
- citation_valid: answer cites at least one expected page and no invented page
- refusal_correct: for unanswerable questions, did it say NOT_FOUND instead of guessing?

The refusal check is the one I care about most. From doing LLM evaluation
work, the worst failure is a confident wrong answer, e.g. inventing a salary
figure that is not in the notification.
"""

import argparse
import json
import tempfile
from pathlib import Path

from saathi.config import settings
from saathi.embeddings import HashEmbedder
from saathi.pipeline import Saathi

HERE = Path(__file__).resolve().parent
SAMPLE_PDF = HERE.parent / "data" / "sample" / "sample_notification.pdf"


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def retrieval_metrics(saathi: Saathi, doc_id: str, questions: list[dict], k: int) -> list[dict]:
    rows = []
    for q in questions:
        if not q["answerable"]:
            continue
        hits = saathi.retriever.retrieve(q["question"], k=k, doc_id=doc_id)
        pages = [h.chunk.page for h in hits]
        first = next((i for i, p in enumerate(pages, 1) if p in q["expected_pages"]), None)
        rows.append({"id": q["id"], "hit": first is not None, "rr": 1 / first if first else 0.0, "pages": pages})
    return rows


def answer_metrics(saathi: Saathi, doc_id: str, questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        ans = saathi.ask(q["question"], doc_id)
        row = {"id": q["id"], "answerable": q["answerable"], "found": ans.found, "answer": ans.text}
        if q["answerable"]:
            text = ans.text.lower()
            row["correct"] = ans.found and all(s.lower() in text for s in q["must_include"])
            row["citation_valid"] = (
                bool(set(ans.cited_pages) & set(q["expected_pages"])) and not ans.invalid_citations
            )
        else:
            row["refusal_correct"] = not ans.found
        rows.append(row)
    return rows


def pct(values) -> str:
    values = list(values)
    return f"{100 * sum(values) / len(values):.0f}%" if values else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--fake-embedder", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default=str(HERE / "results.json"))
    args = parser.parse_args()

    if args.fake_embedder:
        embedder = HashEmbedder()
    else:
        from saathi.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(settings.embedding_model)

    llm = None
    if args.with_llm:
        from saathi.llm import GeminiLLM

        llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model)

    questions = load_questions(HERE / "questions.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        saathi = Saathi(embedder, llm, index_dir=Path(tmp))
        doc_id = saathi.ingest(SAMPLE_PDF)

        ret = retrieval_metrics(saathi, doc_id, questions, args.k)
        summary = {
            "embedder": "hash" if args.fake_embedder else settings.embedding_model,
            "k": args.k,
            f"hit@{args.k}": pct(r["hit"] for r in ret),
            "mrr": round(sum(r["rr"] for r in ret) / len(ret), 3),
        }
        misses = [r for r in ret if not r["hit"]]

        ans_rows = []
        if llm:
            ans_rows = answer_metrics(saathi, doc_id, questions)
            answerable = [r for r in ans_rows if r["answerable"]]
            unanswerable = [r for r in ans_rows if not r["answerable"]]
            summary["answer_correct"] = pct(r["correct"] for r in answerable)
            summary["citation_valid"] = pct(r["citation_valid"] for r in answerable)
            summary["refusal_correct"] = pct(r["refusal_correct"] for r in unanswerable)

    print("\n".join(f"{k:>16}: {v}" for k, v in summary.items()))
    if misses:
        print("\nretrieval misses:", ", ".join(f"{m['id']} (got pages {m['pages']})" for m in misses))
    Path(args.out).write_text(
        json.dumps({"summary": summary, "retrieval": ret, "answers": ans_rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
