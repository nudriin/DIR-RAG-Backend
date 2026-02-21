from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Humbet AI Chatbot Backend"
    environment: Literal["local", "dev", "prod"] = "local"

    rag_mode: Literal["naive", "advanced", "modular"] = Field(
        default="modular", env="RAG_MODE"
    )

    vector_backend: Literal["faiss", "chroma"] = Field(
        default="chroma", env="VECTOR_BACKEND"
    )

    embedding_model: str = Field(
        default="text-embedding-ada-002", env="EMBEDDING_MODEL"
    )

    use_bge: bool = Field(default=False, env="USE_BGE")
    bge_model_name: str = Field(
        default="BAAI/bge-m3", env="BGE_MODEL_NAME"
    )
    gemini_embedding_model: str = Field(
        default="models/text-embedding-004", env="GEMINI_EMBEDDING_MODEL"
    )

    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")
    gpt_model: str = Field(default="gpt-4", env="GPT_MODEL")
    top_logprops: int = Field(default=5, env="TOP_LOGPROBS")

    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    replicate_api_token: str | None = Field(default=None, env="REPLICATE_API_TOKEN")
    hf_token: str | None = Field(default=None, env="HF_TOKEN")
    enable_dragin: bool = Field(default=True, env="ENABLE_DRAGIN")

    database_url: str | None = Field(default=None, env="DATABASE_URL")

    # DRAGIN LLM backend: "openai" atau "gemini"
    dragin_llm_backend: str = Field(default="openai", env="DRAGIN_LLM_BACKEND")
    gemini_model: str = Field(default="gemini-2.0-flash", env="GEMINI_MODEL")
    google_api_key: str | None = Field(default=None, env="GOOGLE_API_KEY")

    max_iterations: int = Field(default=2, env="MAX_ITERATIONS")
    similarity_top_k: int = Field(default=5, env="SIMILARITY_TOP_K")

    jwt_secret_key: str = Field(default="change-me", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_exp_hours: int = Field(default=24, env="ACCESS_TOKEN_EXP_HOURS")
    refresh_token_exp_days: int = Field(default=7, env="REFRESH_TOKEN_EXP_DAYS")

    admin_register_rate_limit_per_hour: int = Field(
        default=5, env="ADMIN_REGISTER_RATE_LIMIT_PER_HOUR"
    )

    cors_allow_origins: List[str] = Field(
        default=["*"], env="CORS_ALLOW_ORIGINS"
    )

    default_admin_password: str | None = Field(
        default=None, env="DEFAULT_ADMIN_PASSWORD"
    )

    dragin_threshold: float = Field(default=0.5, env="DRAGIN_THRESHOLD")
    dragin_max_iterations: int = Field(default=2, env="DRAGIN_MAX_ITERATIONS")
    reranker_min_score: float = Field(default=0.1, env="RERANKER_MIN_SCORE")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        env="RERANKER_MODEL",
    )

    semantic_breakpoint_threshold_amount: int = Field(
        default=80, env="SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT"
    )

    # Context budgeting to avoid rate limit/token overflow
    context_max_docs: int = Field(default=5, env="CONTEXT_MAX_DOCS")
    context_char_budget: int = Field(default=6000, env="CONTEXT_CHAR_BUDGET")
    max_generation_tokens: int = Field(default=512, env="MAX_GENERATION_TOKENS")
    rq_similarity_block_threshold: float = Field(
        default=0.7, env="RQ_SIMILARITY_BLOCK_THRESHOLD"
    )
    rq_bypass_confidence_threshold: float = Field(
        default=0.7, env="RQ_BYPASS_CONFIDENCE_THRESHOLD"
    )
    rq_postvalidate_margin: float = Field(
        default=0.15, env="RQ_POSTVALIDATE_MARGIN"
    )
    rq_concentration_scale: float = Field(
        default=0.2, env="RQ_CONCENTRATION_SCALE"
    )
    rq_length_norm: int = Field(
        default=6, env="RQ_LENGTH_NORM"
    )
    rq_enable_bypass: bool = Field(
        default=True, env="RQ_ENABLE_BYPASS"
    )

    base_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = base_dir / "storage"
    vector_dir: Path = data_dir / "vectors"
    log_dir: Path = data_dir / "logs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.vector_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    return settings
