"""Index one PDF or a whole folder from the command line.

    python -m scripts.ingest data/notifications/
"""

import argparse
import logging
from pathlib import Path

from saathi.config import settings
from saathi.embeddings import make_embedder
from saathi.pipeline import Saathi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="PDF file or folder of PDFs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    path = Path(args.path)
    pdfs = sorted(path.glob("*.pdf")) if path.is_dir() else [path]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {path}")

    embedder = make_embedder(settings.embedding_provider, settings.embedding_model, settings.gemini_api_key)
    saathi = Saathi(embedder, llm=None)
    for pdf in pdfs:
        try:
            print(pdf.name, "->", saathi.ingest(pdf))
        except ValueError as err:
            print("skipped", pdf.name, ":", err)


if __name__ == "__main__":
    main()
