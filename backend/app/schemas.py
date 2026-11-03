"""Public API contracts (request/response bodies)."""

from typing import Literal

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: str
    kind: Literal["resume", "job"]
    label: str
    title: str
    filename: str
    chunks: int
    chars: int
    warnings: list[str] = []


class SessionState(BaseModel):
    session_id: str
    resume: DocumentInfo | None
    jobs: list[DocumentInfo]
    llm_enabled: bool


class TextDocumentIn(BaseModel):
    title: str = Field(default="", max_length=120)
    text: str = Field(min_length=1, max_length=200_000)


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class Source(BaseModel):
    id: str
    label: str
    title: str
    section: str
    text: str
    cosine: float


# --- fit report -------------------------------------------------------------------------


class MatchedSkill(BaseModel):
    skill: str
    evidence: str
    verified: bool = False  # evidence quote was found verbatim in the resume


class MissingSkill(BaseModel):
    skill: str
    importance: Literal["required", "preferred"] = "preferred"
    suggestion: str = ""


class InterviewQuestion(BaseModel):
    question: str
    talking_point: str = ""


class FitReport(BaseModel):
    job_id: str
    job_label: str
    job_title: str
    match_score: int = Field(ge=0, le=100)
    summary: str
    matched_skills: list[MatchedSkill]
    missing_skills: list[MissingSkill]
    experience_alignment: str
    interview_questions: list[InterviewQuestion]


class LLMFitOutput(BaseModel):
    """What we accept from the model. Lenient on shape, strict on types."""

    match_score: int = Field(ge=0, le=100)
    summary: str = ""
    matched_skills: list[MatchedSkill] = []
    missing_skills: list[MissingSkill] = []
    experience_alignment: str = ""
    interview_questions: list[InterviewQuestion] = []
