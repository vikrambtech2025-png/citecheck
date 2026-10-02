"""Central configuration. All tunables live here so runs are reproducible."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CITECHECK_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model backend -----------------------------------------------------
    # One of: "ollama" | "openai_compat" | "openai" | "anthropic" | "mock"
    model_backend: str = "ollama"

    # Local Ollama. OLLAMA_NUM_CTX is read from the environment by the daemon,
    # not here -- see README "Hardware notes".
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b-instruct"

    # Any OpenAI-compatible endpoint (vLLM, LM Studio, Together, Groq, ...)
    openai_compat_base_url: str = "http://localhost:8000/v1"
    openai_compat_model: str = "local-model"
    openai_compat_api_key: str = "not-needed"

    # Hosted providers (optional; only used when you deliberately switch).
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"

    # --- Sampling ----------------------------------------------------------
    temperature: float = 0.0
    max_output_tokens: int = 2048
    request_timeout_s: float = 120.0
    max_retries: int = 3
    retry_backoff_s: float = 1.5

    # --- Agent budgets (guardrail: the agent cannot exceed these) ----------
    max_steps: int = 12
    max_tool_calls: int = 40
    max_tokens_total: int = 120_000
    max_wall_clock_s: float = 300.0

    # --- Filesystem sandbox ------------------------------------------------
    # Every file tool is confined to this directory. Absolute and traversal
    # paths are resolved and rejected at the code layer, not in the prompt.
    workspace_dir: Path = Field(default_factory=lambda: REPO_ROOT / "workspace")
    outbox_dir: Path = Field(default_factory=lambda: REPO_ROOT / "outbox")
    max_file_bytes: int = 2_000_000

    # --- Web tool limits ---------------------------------------------------
    max_fetch_bytes: int = 3_000_000
    fetch_timeout_s: float = 20.0
    allow_private_network: bool = False  # SSRF guard; keep False
    search_backend: str = "duckduckgo"  # "duckduckgo" | "wikipedia" | "off"

    # --- SQLite tool -------------------------------------------------------
    sqlite_path: Path | None = None  # defaults to workspace/knowledge.sqlite3
    sqlite_max_rows: int = 200
    sqlite_query_timeout_s: float = 5.0

    # --- Tracing -----------------------------------------------------------
    tracing_backend: str = "none"  # "none" | "langfuse" | "otel"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    otel_otlp_endpoint: str = ""

    # --- Storage -----------------------------------------------------------
    state_db_path: Path = Field(default_factory=lambda: REPO_ROOT / "runs" / "state.sqlite3")

    def ensure_dirs(self) -> None:
        for d in (
            self.workspace_dir,
            self.outbox_dir,
            self.state_db_path.parent,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s