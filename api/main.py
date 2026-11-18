"""REST API.

    uvicorn api.main:app --reload
    open http://127.0.0.1:8000/docs
"""

import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from saathi.pipeline import Saathi
from saathi.schemas import NotificationDetails, PostEligibility, UserProfile

app = FastAPI(title="Sarkari Saathi API", version="0.1.0")


@lru_cache
def get_saathi() -> Saathi:
    return Saathi.from_settings()


class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None


class Source(BaseModel):
    page: int
    text: str


class AskResponse(BaseModel):
    answer: str
    found: bool
    cited_pages: list[int]
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents(saathi: Saathi = Depends(get_saathi)):
    return {"documents": saathi.documents()}


@app.post("/documents")
def upload(file: UploadFile = File(...), saathi: Saathi = Depends(get_saathi)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(file.filename).name
        with path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            doc_id = saathi.ingest(path)
        except ValueError as err:
            raise HTTPException(422, str(err))
    return {"doc_id": doc_id}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, saathi: Saathi = Depends(get_saathi)):
    if not req.question.strip():
        raise HTTPException(400, "Question is empty")
    ans = saathi.ask(req.question, req.doc_id)
    return AskResponse(
        answer=ans.text,
        found=ans.found,
        cited_pages=ans.cited_pages,
        sources=[Source(page=s.chunk.page, text=s.chunk.text) for s in ans.sources],
    )


@app.get("/documents/{doc_id}/details", response_model=NotificationDetails)
def details(doc_id: str, saathi: Saathi = Depends(get_saathi)):
    if doc_id not in saathi.documents():
        raise HTTPException(404, "Unknown document")
    return saathi.details(doc_id)


@app.post("/documents/{doc_id}/eligibility", response_model=list[PostEligibility])
def eligibility(doc_id: str, profile: UserProfile, saathi: Saathi = Depends(get_saathi)):
    if doc_id not in saathi.documents():
        raise HTTPException(404, "Unknown document")
    return saathi.eligibility(doc_id, profile)
