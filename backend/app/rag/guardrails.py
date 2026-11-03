"""Input and output guardrails. Deliberately simple, deterministic and unit-tested.

These are *defence in depth*, not a guarantee: the primary defence against prompt injection is
structural (untrusted text is fenced in delimiters, the model is told it is data, the model has
no tools and cannot take actions, and its output is only rendered as plain text/markdown).
"""

import re
from dataclasses import dataclass

from app.errors import GuardrailViolation

# Phrases that attempt to hijack instructions. Matched case-insensitively.
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above|earlier|your) (instructions|prompts?|rules)",
    r"forget (all |everything|your) (previous |prior )?(instructions|rules)",
    r"you are now (?!a candidate)",
    r"(reveal|print|show|repeat|output) (me )?(your|the) (system|hidden|initial) (prompt|instructions?)",
    r"new (system )?instructions?:",
    r"(act|behave) as if you (are|were) (?!a candidate)",
    r"pretend (to be|you are|you're) (?!a candidate)",
    r"</?(system|assistant|context|document)>",
    r"\bjailbreak\b|\bDAN mode\b",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)

_CITATION = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def check_question(question: str, max_chars: int) -> str:
    """Validate a user question. Returns the cleaned question or raises GuardrailViolation."""
    q = question.strip()
    if not q:
        raise GuardrailViolation("empty", "Please type a question.")
    if len(q) > max_chars:
        raise GuardrailViolation("too_long", f"Questions are limited to {max_chars} characters.")
    if _INJECTION_RE.search(q):
        raise GuardrailViolation(
            "prompt_injection",
            "I can only help with questions about your resume and the job postings you uploaded.",
        )
    return q


def scan_document_for_injection(text: str) -> list[str]:
    """Uploaded documents are untrusted. We don't reject them (a resume may legitimately contain
    odd text) but we surface a warning so the user knows, and the prompt fences them as data."""
    if _INJECTION_RE.search(text):
        return ["This document contains text that looks like instructions to an AI. It will be treated as plain content and ignored as instructions."]  # fmt: skip
    return []


@dataclass(frozen=True)
class CitationCheck:
    cited: list[str]  # valid source ids, in order of first appearance
    invalid: list[str]  # ids the model made up
    has_any: bool


def check_citations(answer: str, valid_ids: set[str]) -> CitationCheck:
    """Verify every [S#] tag the model emitted refers to a passage we actually gave it."""
    seen: list[str] = []
    invalid: list[str] = []
    for group in _CITATION.findall(answer):
        for sid in re.split(r"\s*,\s*", group):
            if sid in valid_ids:
                if sid not in seen:
                    seen.append(sid)
            elif sid not in invalid:
                invalid.append(sid)
    return CitationCheck(cited=seen, invalid=invalid, has_any=bool(seen or invalid))


def strip_invalid_citations(answer: str, invalid: list[str]) -> str:
    out = answer
    for sid in invalid:
        out = re.sub(rf"\s*\[{sid}\]", "", out)
        out = re.sub(rf"\b{sid}\s*,\s*|\s*,\s*{sid}\b", "", out)
    return out


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip()


def quote_in_document(quote: str, document_text: str) -> bool:
    """True if `quote` appears (modulo case/punctuation/whitespace) in the document."""
    q = normalize_for_match(quote)
    return len(q) >= 3 and q in normalize_for_match(document_text)
