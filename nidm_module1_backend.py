"""
NIDM - Module 1 Backend Starter (FastAPI + SQLModel)
Single-file starter for PDF upload, extraction, metadata store, and basic retrieval.

Files/DB created locally for easy start. Replace Postgres URL in DATABASE_URL for production.

How to run (local dev):
1. Create virtualenv: python -m venv .venv && source .venv/bin/activate
2. Install requirements: pip install -r requirements.txt
3. Run: uvicorn nidm_module1_backend:app --reload --port 8000

Notes:
- This is a starter template. It uses SQLModel (works with PostgreSQL or SQLite) for simplicity.
- Extraction: PyMuPDF (fitz) used for text extraction. If PDF is scanned, pytesseract/OCR fallback is included.
- Background processing: FastAPI BackgroundTasks is used to process PDFs asynchronously at upload time.


Project structure (single-file starter):
- nidm_module1_backend.py  <-- this file
- uploads/                 <-- PDF files saved here
- data/                    <-- sqlite DB file (if using sqlite)

Requirements (requirements.txt):
fastapi
uvicorn[standard]
sqlmodel
pymupdf
pillow
pytesseract
python-multipart
python-dotenv

You can copy the following into requirements.txt.

"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, Session, create_engine, select
from typing import Optional, List
import uuid
import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io
import datetime
import shutil
import re

# ---------------------------
# Configuration
# ---------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/nidm_module1.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_URL.replace('sqlite:///', '')) or './data', exist_ok=True)

engine = create_engine(DATABASE_URL, echo=False)
app = FastAPI(title="NIDM Module1 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Database models
# ---------------------------
class Article(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: str = Field(index=True, default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    authors: Optional[str] = None  # store as comma-separated for simplicity
    year: Optional[int] = None
    source: Optional[str] = None
    tags: Optional[str] = None  # comma-separated
    pdf_path: Optional[str] = None
    uploaded_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)


class ExtractedText(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    article_id: int = Field(foreign_key="article.id")
    section: Optional[str] = None
    content: Optional[str] = None


# ---------------------------
# DB init
# ---------------------------
def init_db():
    SQLModel.metadata.create_all(engine)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------
# PDF extraction utilities
# ---------------------------

def save_upload_file(upload_file: UploadFile, destination: str) -> None:
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def extract_text_from_pdf(path: str) -> str:
    """Extracts text from PDF using PyMuPDF. If no text found, tries OCR on pages."""
    doc = fitz.open(path)
    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text and text.strip():
            full_text.append(text)
        else:
            # Fallback to OCR: render page to image and run pytesseract
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes()
            img = Image.open(io.BytesIO(img_bytes))
            try:
                ocr_text = pytesseract.image_to_string(img)
                full_text.append(ocr_text)
            except Exception:
                full_text.append("")
    return "\n".join(full_text)


def clean_text(text: str) -> str:
    # Basic cleaning: remove excessive whitespace, headers/footers heuristics
    # Remove multiple newlines
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Remove repeated page headers like 'Page X of Y' or similar
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
    # Trim
    return text.strip()


def simple_section_split(text: str) -> List[dict]:
    """Naive section splitter that looks for common headings.
    Returns list of {section: name, content: content}
    """
    headings = ["abstract", "introduction", "method", "methodology", "materials", "results", "discussion", "conclusion", "references"]
    # Normalize
    lines = text.splitlines()
    sections = []
    current_section = "body"
    buffer = []
    for line in lines:
        lline = line.strip().lower()
        if any(lline.startswith(h) for h in headings):
            # flush existing
            if buffer:
                sections.append({"section": current_section, "content": "\n".join(buffer).strip()})
            current_section = lline.split()[0]
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        sections.append({"section": current_section, "content": "\n".join(buffer).strip()})
    return sections


# ---------------------------
# Background processing
# ---------------------------

def process_pdf_and_store(article_id: int, pdf_path: str):
    try:
        raw = extract_text_from_pdf(pdf_path)
        cleaned = clean_text(raw)
        sections = simple_section_split(cleaned)
        with Session(engine) as session:
            # Save sections
            for sec in sections:
                et = ExtractedText(article_id=article_id, section=sec.get("section"), content=sec.get("content"))
                session.add(et)
            session.commit()
            # Update title/authors heuristics from first lines
            first_chunk = cleaned.splitlines()[:10]
            maybe_title = " ".join(first_chunk[:3]).strip()
            article = session.get(Article, article_id)
            if article and (not article.title):
                article.title = maybe_title[:300]
                session.add(article)
                session.commit()
    except Exception as e:
        print("Error processing PDF:", e)


# ---------------------------
# API Endpoints
# ---------------------------

@app.post("/upload_pdf")
def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...), title: Optional[str] = None, authors: Optional[str] = None, year: Optional[int] = None, source: Optional[str] = None):
    # Basic validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    filename = f"{uuid.uuid4()}_{os.path.basename(file.filename)}"
    dest_path = os.path.join(UPLOAD_DIR, filename)
    try:
        save_upload_file(file, dest_path)
    finally:
        file.file.close()

    # Create DB record
    article = Article(title=title, authors=authors, year=year, source=source, pdf_path=dest_path)
    with Session(engine) as session:
        session.add(article)
        session.commit()
        session.refresh(article)
        article_id = article.id

    # Process in background
    background_tasks.add_task(process_pdf_and_store, article_id, dest_path)

    return JSONResponse({"status": "processing", "article_id": article_id, "uuid": article.uuid})


@app.get("/articles")
def list_articles(limit: int = 20, offset: int = 0):
    with Session(engine) as session:
        q = session.exec(select(Article).offset(offset).limit(limit))
        rows = q.all()
        result = []
        for r in rows:
            result.append({
                "id": r.id,
                "uuid": r.uuid,
                "title": r.title,
                "authors": r.authors,
                "year": r.year,
                "source": r.source,
                "tags": r.tags,
                "uploaded_at": r.uploaded_at.isoformat(),
            })
        return result


@app.get("/articles/{article_id}")
def get_article(article_id: int):
    with Session(engine) as session:
        article = session.get(Article, article_id)
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        # fetch extracted text summary (first 300 chars of each section)
        q = session.exec(select(ExtractedText).where(ExtractedText.article_id == article_id))
        secs = q.all()
        sections = [{"section": s.section, "preview": (s.content[:300] + '...') if s.content and len(s.content) > 300 else s.content} for s in secs]
        return {
            "id": article.id,
            "uuid": article.uuid,
            "title": article.title,
            "authors": article.authors,
            "year": article.year,
            "source": article.source,
            "tags": article.tags,
            "uploaded_at": article.uploaded_at.isoformat(),
            "sections": sections,
        }


@app.get("/articles/{article_id}/sections/{section_name}")
def get_section(article_id: int, section_name: str):
    with Session(engine) as session:
        q = session.exec(select(ExtractedText).where(ExtractedText.article_id == article_id).where(ExtractedText.section.ilike(f"%{section_name}%")))
        secs = q.all()
        if not secs:
            raise HTTPException(status_code=404, detail="Section not found")
        return {"sections": [{"id": s.id, "section": s.section, "content": s.content} for s in secs]}


# Healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nidm_module1_backend:app", host="0.0.0.0", port=8000, reload=True)
