"""Application settings, loaded from environment variables / .env (never hardcoded)."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore", populate_by_name=True)

    # --- server ---
    # "127.0.0.1" locally. "container" = bind to the container's own interface IP
    # (never 0.0.0.0), see app/main.py.
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    static_dir: Path | None = Path("static")
    samples_dir: Path = Path("../samples")  # demo documents for the "Try the demo" button

    # --- LLM (any OpenAI-compatible endpoint: OpenAI, Azure OpenAI, Ollama, vLLM...) ---
    llm_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY")
    )
    llm_base_url: str | None = None
    llm_model: str = "gpt-4.1-mini"
    llm_timeout_s: float = 60.0
    llm_temperature: float = 0.2

    # --- embeddings (local ONNX via fastembed: no data leaves the machine) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- ingestion limits ---
    max_upload_bytes: int = 5 * 1024 * 1024
    max_document_chars: int = 60_000
    min_document_chars: int = 200
    max_jobs_per_session: int = 5
    chunk_max_chars: int = 900

    # --- retrieval / context budget ---
    resume_top_k: int = 5
    jobs_total_top_k: int = 8
    max_context_chars: int = 12_000
    max_history_turns: int = 6
    max_question_chars: int = 1_000

    # --- sessions (in-memory, ephemeral: documents contain PII) ---
    session_ttl_s: int = 2 * 60 * 60
    max_sessions: int = 200

    @property
    def llm_enabled(self) -> bool:
        return self.llm_api_key is not None or self.llm_base_url is not None


@lru_cache
def get_settings() -> Settings:
    return Settings()
