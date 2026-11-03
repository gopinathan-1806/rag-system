"""Turn uploaded bytes into clean text. Supports PDF, DOCX, TXT and MD."""

import io
import re
from pathlib import PurePath

from docx import Document as DocxDocument
from pypdf import PdfReader

from app.errors import ParseError, PayloadTooLargeError

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
_PDF_MAGIC = b"%PDF-"
_DOCX_MAGIC = b"PK\x03\x04"  # docx is a zip container
_MAX_PDF_PAGES = 15


def parse_document(filename: str, data: bytes, *, max_bytes: int, max_chars: int) -> str:
    """Validate and extract text. Raises ParseError / PayloadTooLargeError with safe messages."""
    if len(data) > max_bytes:
        raise PayloadTooLargeError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    if not data:
        raise ParseError("The file is empty.")

    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParseError("Unsupported file type. Please upload PDF, DOCX, TXT or MD.")

    # Check content, not just the extension, so a renamed binary is rejected.
    if ext == ".pdf":
        text = _read_pdf(data)
    elif ext == ".docx":
        text = _read_docx(data)
    else:
        text = _read_text(data)

    text = normalize_text(text)
    if len(text) > max_chars:
        raise PayloadTooLargeError(f"Document is too long ({len(text):,} characters; limit {max_chars:,}).")
    return text


def _read_pdf(data: bytes) -> str:
    if not data.startswith(_PDF_MAGIC):
        raise ParseError("The file does not look like a valid PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ParseError("Password-protected PDFs are not supported.")
        if len(reader.pages) > _MAX_PDF_PAGES:
            raise ParseError(f"PDF has too many pages (limit {_MAX_PDF_PAGES}).")
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ParseError:
        raise
    except Exception as exc:  # pypdf raises many exception types on malformed input
        raise ParseError("Could not read this PDF.") from exc


def _read_docx(data: bytes) -> str:
    if not data.startswith(_DOCX_MAGIC):
        raise ParseError("The file does not look like a valid DOCX.")
    try:
        doc = DocxDocument(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(parts)
    except Exception as exc:
        raise ParseError("Could not read this DOCX.") from exc


def _read_text(data: bytes) -> str:
    if b"\x00" in data[:4096]:
        raise ParseError("The file looks binary, not text.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


_BULLETS = re.compile(r"^[ \t]*[•●▪◦‣·*–-][ \t]+", re.MULTILINE)


def normalize_text(text: str) -> str:
    """Unify bullets, strip control chars and collapse whitespace while keeping line structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = _BULLETS.sub("- ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
