from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    app_name: str = "Humbet AI Chatbot Backend"
    environment: Literal["local", "dev", "prod"] = "local"

    rag_mode: Literal["naive", "advanced", "modular"] = Field(
        default="modular", env="RAG_MODE"
    )

    vector_backend: Literal["faiss", "chroma"] = Field(
        default="faiss", env="VECTOR_BACKEND"
    )

    embedding_model: str = Field(
        default="text-embedding-ada-002", env="EMBEDDING_MODEL"
    )

    use_bge: bool = Field(default=False, env="USE_BGE")

    llm_model: str = Field(default="gpt-4", env="LLM_MODEL")

    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")

    max_iterations: int = Field(default=3, env="MAX_ITERATIONS")
    similarity_top_k: int = Field(default=5, env="SIMILARITY_TOP_K")

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

