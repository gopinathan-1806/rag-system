"""Core domain objects (plain dataclasses, independent of the web layer)."""

from dataclasses import dataclass, field
from typing import Literal

DocKind = Literal["resume", "job"]


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    doc_label: str  # "Resume" or "Job #2"
    doc_title: str
    section: str
    text: str

    @property
    def embedding_text(self) -> str:
        """Text that is embedded: body plus a header so the vector knows where it came from."""
        return f"{self.doc_label} ({self.doc_title}) - {self.section}\n{self.text}"


@dataclass
class Document:
    id: str
    kind: DocKind
    filename: str
    title: str
    number: int | None  # job number, stable for the lifetime of the session
    text: str
    chunks: list[Chunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "Resume" if self.kind == "resume" else f"Job #{self.number}"


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str
