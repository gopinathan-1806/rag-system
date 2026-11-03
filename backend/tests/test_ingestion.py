import io

import pytest
from docx import Document as Docx

from app.errors import ParseError, PayloadTooLargeError
from app.ingestion.chunking import chunk_document, split_sections
from app.ingestion.parsing import normalize_text, parse_document
from app.models import Document

from .conftest import sample

LIMITS = {"max_bytes": 1_000_000, "max_chars": 60_000}


def test_parses_txt_and_normalizes_bullets() -> None:
    text = parse_document("a.txt", "• one\r\n●  two\n\n\n\nthree".encode(), **LIMITS)
    assert text == "- one\n- two\n\nthree"


def test_parses_docx() -> None:
    buf = io.BytesIO()
    d = Docx()
    d.add_paragraph("Hello docx")
    d.save(buf)
    assert "Hello docx" in parse_document("r.docx", buf.getvalue(), **LIMITS)


@pytest.mark.parametrize(
    ("name", "data", "exc"),
    [
        ("a.exe", b"x", ParseError),
        ("a.pdf", b"not a pdf", ParseError),
        ("a.docx", b"not a zip", ParseError),
        ("a.txt", b"\x00\x01\x02binary", ParseError),
        ("a.txt", b"", ParseError),
        ("a.txt", b"x" * 2_000_000, PayloadTooLargeError),
    ],
)
def test_rejects_bad_files(name: str, data: bytes, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        parse_document(name, data, **LIMITS)


def test_rejects_too_many_characters() -> None:
    with pytest.raises(PayloadTooLargeError):
        parse_document("a.txt", b"x" * 500, max_bytes=1000, max_chars=100)


def test_normalize_strips_control_chars() -> None:
    assert normalize_text("a\x00b\x07c") == "abc"


def _doc(text: str) -> Document:
    return Document(id="d1", kind="job", filename="j.txt", title="T", number=2, text=text)


def test_sections_detected_for_sample_job() -> None:
    names = [n for n, _ in split_sections(sample("job_ml_search_engineer.txt").decode())]
    assert {"Responsibilities", "Requirements", "Nice To Have"} <= set(names)


def test_chunks_never_cross_sections_and_carry_metadata() -> None:
    doc = _doc(sample("job_ml_search_engineer.txt").decode())
    chunks = chunk_document(doc, max_chars=300)
    assert chunks
    req = [c for c in chunks if c.section == "Requirements"]
    assert req and all("Responsibilities" not in c.text for c in req)
    assert all(c.doc_label == "Job #2" and len(c.text) <= 300 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_oversized_paragraph_is_split() -> None:
    doc = _doc("SUMMARY\n" + "Sentence number one is here. " * 100)
    chunks = chunk_document(doc, max_chars=200)
    assert len(chunks) > 5 and all(len(c.text) <= 200 for c in chunks)


def test_no_punctuation_blob_is_split() -> None:
    doc = _doc("x" * 1000)
    assert all(len(c.text) <= 100 for c in chunk_document(doc, max_chars=100))
