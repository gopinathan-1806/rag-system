"""Section-aware chunking for resumes and job descriptions.

Why not fixed-size windows? Both document types are short and highly structured. A fixed
window happily cuts "Requirements" in half or glues "Skills" onto the next job's bullets.
We split on detected headings first, then pack paragraphs/bullets inside a section up to a
size budget. Every chunk keeps its section name, which is later embedded and shown as a citation.
"""

import re
from collections.abc import Iterator

from app.models import Chunk, DocKind, Document

_KNOWN_HEADINGS = {
    # resume
    "summary", "profile", "objective", "about", "about me", "experience",
    "work experience", "professional experience", "employment", "employment history",
    "education", "skills", "technical skills", "core skills", "projects", "certifications",
    "certificates", "publications", "awards", "achievements", "languages", "volunteering",
    "interests",
    # job posting
    "responsibilities", "requirements", "qualifications", "minimum qualifications",
    "preferred qualifications", "nice to have", "what you'll do", "what you will do",
    "what we're looking for", "who you are", "about the role", "about the job",
    "about the team", "about us", "benefits", "perks", "what we offer", "the role",
    "key responsibilities", "required skills", "preferred skills", "bonus points",
}  # fmt: skip

_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
_BULLET = re.compile(r"^- ")


def _heading_of(line: str) -> str | None:
    """Return the heading text if `line` looks like a section heading, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None
    if m := _MD_HEADING.match(stripped):
        return m.group(1).strip().rstrip(":")
    bare = stripped.rstrip(":").strip().lower()
    if bare in _KNOWN_HEADINGS:
        return stripped.rstrip(":").strip().title()
    # ALL-CAPS short lines ("WORK EXPERIENCE") that are not sentences
    letters = [c for c in stripped if c.isalpha()]
    if (
        len(letters) >= 4
        and all(c.isupper() for c in letters)
        and not stripped.endswith((".", ","))
        and len(stripped.split()) <= 5
    ):
        return stripped.rstrip(":").title()
    return None


def extract_title(text: str, kind: DocKind, filename: str) -> str:
    """Best-effort human title: first non-heading line of a job, the person's name for a resume."""
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line and _heading_of(line) is None and len(line) <= 90:
            return line
    return filename.rsplit(".", 1)[0]


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(section_name, body)]. Text before the first heading is the 'Header' section."""
    sections: list[tuple[str, list[str]]] = [("Header", [])]
    for line in text.splitlines():
        heading = _heading_of(line)
        if heading is not None:
            sections.append((heading, []))
        else:
            sections[-1][1].append(line)
    return [(name, "\n".join(body).strip()) for name, body in sections if "".join(body).strip()]


def _blocks(body: str) -> Iterator[str]:
    """Atomic units inside a section: paragraphs, with bullet-lists kept line by line."""
    for para in re.split(r"\n\s*\n", body):
        lines = [ln for ln in para.split("\n") if ln.strip()]
        if lines and sum(bool(_BULLET.match(ln)) for ln in lines) >= max(1, len(lines) // 2):
            yield from lines  # bullets are individually meaningful
        elif lines:
            yield "\n".join(lines)


def _pack(blocks: Iterator[str], max_chars: int) -> Iterator[str]:
    current = ""
    for block in blocks:
        # A single oversized block (e.g. a PDF paragraph with no newlines): hard split on sentences.
        pieces = [block] if len(block) <= max_chars else _split_long(block, max_chars)
        for piece in pieces:
            if current and len(current) + len(piece) + 1 > max_chars:
                yield current
                current = ""
            current = f"{current}\n{piece}" if current else piece
    if current:
        yield current


def _split_long(block: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", block)
    out: list[str] = []
    cur = ""
    for s in sentences:
        while len(s) > max_chars:  # pathological: no punctuation at all
            out.append(s[:max_chars])
            s = s[max_chars:]
        if cur and len(cur) + len(s) + 1 > max_chars:
            out.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_document(doc: Document, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in split_sections(doc.text):
        for piece in _pack(_blocks(body), max_chars):
            chunks.append(
                Chunk(
                    id=f"{doc.id}:{len(chunks)}",
                    doc_id=doc.id,
                    doc_label=doc.label,
                    doc_title=doc.title,
                    section=section,
                    text=piece,
                )
            )
    return chunks
