"""Application configuration and canonical paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"), extra="ignore", case_sensitive=False
    )

    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    database_url: str = f"sqlite:///{(REPO_ROOT / 'artifacts' / 'insightdesk.db').as_posix()}"

    # --- LLM (optional; the app is fully functional with provider "none") ---
    llm_provider: str = "none"
    llm_model: str = "claude-opus-5"
    # Base URL for an OpenAI-compatible provider. Left blank, a known provider
    # name (groq, gemini, cerebras, openrouter, together, mistral) supplies its
    # default; set this to override or to point at anything else.
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_timeout_seconds: int = 60
    # Caps thinking AND response text together on Opus 5, where thinking is on
    # by default. Too small truncates the JSON and forces a pointless fallback.
    llm_max_output_tokens: int = 8000
    llm_max_evidence_cases: int = 5
    llm_max_evidence_chars: int = 6000

    # --- embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str | None = None
    embedding_batch_size: int = 256

    # --- observability (Phase 10) ---
    # Console exporter only. No SaaS account, no external dependency, and
    # nothing on a resume that cannot be demonstrated locally.
    otel_enabled: bool = False
    otel_exporter: str = "console"

    # --- deployment capacity (Phase 12) --------------------------------------
    # A memory-constrained host may serve a stratified sample of a corpus that
    # was audited in full. When this is set, the manifest discloses BOTH numbers
    # rather than quietly reporting the smaller one as the dataset size.
    corpus_serve_limit: int | None = None

    # --- API ---
    cors_origins: str = "http://localhost:5173"
    max_request_body_bytes: int = 65_536
    max_issue_text_chars: int = 8_000

    # --- retrieval gate thresholds -------------------------------------------
    # UNCALIBRATED conservative defaults for
    # sentence-transformers/all-MiniLM-L6-v2 with cosine similarity on
    # normalized embeddings. Replace only after running Phase 5 against the
    # manually labeled eval set, and update `retrieval_gate_calibrated` when you
    # do. Changing the embedding model invalidates every number below.
    retrieval_gate_calibrated: bool = False
    strong_min_cosine: float = 0.62
    strong_min_margin: float = 0.04
    strong_min_candidates: float = 2
    # On template-heavy corpora many near-identical cases sit within a few
    # thousandths of each other, which flattens both the top-1 margin and the
    # dense/BM25 top-10 overlap even when the evidence is unambiguous. This is
    # the third path to "strong": several independent cases all clearing the
    # strong cosine bar.
    strong_min_consensus: int = 3
    mixed_min_cosine: float = 0.45
    candidate_floor_cosine: float = 0.40
    agreement_top_k: int = 10

    # --- retrieval fusion ---
    rrf_k: int = 60
    # 0.25 * (1/(rrf_k+1)) -- deliberately a quarter of one RRF unit so metadata
    # can break ties without ever outranking genuine relevance.
    metadata_boost_fraction: float = 0.25
    metadata_boost_enabled: bool = False
    dense_candidates: int = 50
    lexical_candidates: int = 50

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def evaluation_dir(self) -> Path:
        return self.data_dir / "evaluation"

    @property
    def tickets_parquet(self) -> Path:
        return self.processed_dir / "tickets.parquet"

    @property
    def audit_json(self) -> Path:
        return self.artifacts_dir / "data_audit.json"

    @property
    def audit_md(self) -> Path:
        return self.artifacts_dir / "data_audit.md"

    @property
    def data_card_json(self) -> Path:
        return self.artifacts_dir / "data_card.json"

    @property
    def capabilities_json(self) -> Path:
        return self.artifacts_dir / "capabilities.json"

    @property
    def retrieval_dir(self) -> Path:
        return self.artifacts_dir / "retrieval"

    @property
    def clustering_dir(self) -> Path:
        return self.artifacts_dir / "clustering"

    @property
    def risk_dir(self) -> Path:
        return self.artifacts_dir / "risk"

    @property
    def embeddings_dir(self) -> Path:
        return self.artifacts_dir / "embeddings"

    def ensure_dirs(self) -> None:
        for p in (
            self.artifacts_dir,
            self.processed_dir,
            self.evaluation_dir,
            self.retrieval_dir,
            self.clustering_dir,
            self.risk_dir,
            self.embeddings_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
