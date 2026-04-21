"""
gemini_client.py — Singleton factory untuk inisialisasi Google Gemini SDK.

Mendukung dua mode otentikasi:
  - "api_key"   : Google AI Studio (GOOGLE_API_KEY)
  - "vertex_ai" : Google Vertex AI via Service Account JSON

Best practices:
  - Satu titik inisialisasi SDK (separation of concerns).
  - Lazy init: SDK dikonfigurasi pertama kali saat dibutuhkan.
  - Validasi mode dan credential sebelum konfigurasi.
  - Logging jelas agar mudah di-debug di production.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (thread-safe)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_configured_mode: Optional[str] = None  # track mode saat ini agar tidak re-init


def _sa_json_path() -> Optional[Path]:
    """Kembalikan path file Service Account JSON jika ada."""
    from app.core.config import get_settings
    settings = get_settings()
    # Prioritas 1: env var GOOGLE_SERVICE_ACCOUNT_PATH
    if settings.google_service_account_path:
        p = Path(settings.google_service_account_path)
        if p.exists():
            return p
    # Prioritas 2: file default hasil upload dari dashboard
    default = settings.data_dir / "service_accounts" / "gemini_sa.json"
    if default.exists():
        return default
    return None


def configure_genai(force: bool = False) -> str:
    """
    Inisialisasi SDK Gemini sesuai GEMINI_MODE di settings.

    Args:
        force: Paksa re-init meski sudah pernah dikonfigurasi.

    Returns:
        Mode yang aktif: "api_key" atau "vertex_ai"

    Raises:
        RuntimeError: Jika kredensial tidak tersedia.
    """
    global _configured_mode

    from app.core.config import get_settings
    settings = get_settings()
    mode = (settings.gemini_mode or "api_key").lower()

    with _lock:
        if _configured_mode == mode and not force:
            return mode

        if mode == "vertex_ai":
            _configure_vertex_ai(settings)
        else:
            _configure_api_key(settings)

        _configured_mode = mode
        return mode


def _configure_api_key(settings) -> None:
    """Konfigurasi Gemini via Google AI Studio API Key."""
    api_key = settings.google_api_key
    if not api_key:
        raise RuntimeError(
            "GEMINI_MODE=api_key tetapi GOOGLE_API_KEY tidak ditemukan di environment. "
            "Tambahkan GOOGLE_API_KEY ke file .env."
        )
    genai.configure(api_key=api_key)
    logger.info("Gemini SDK configured via api_key (Google AI Studio)")


def _configure_vertex_ai(settings) -> None:
    """Konfigurasi Gemini via Vertex AI Service Account."""
    sa_path = _sa_json_path()
    if sa_path is None:
        raise RuntimeError(
            "GEMINI_MODE=vertex_ai tetapi file Service Account JSON tidak ditemukan. "
            "Upload SA JSON melalui halaman Settings atau set GOOGLE_SERVICE_ACCOUNT_PATH di .env."
        )

    project = settings.vertex_project
    location = settings.vertex_location or "us-central1"

    # Baca project dari SA JSON jika tidak ada di config
    if not project:
        try:
            with open(sa_path, "r", encoding="utf-8") as f:
                sa_data = json.load(f)
            project = sa_data.get("project_id", "")
        except Exception as exc:
            raise RuntimeError(f"Gagal membaca project_id dari SA JSON: {exc}") from exc

    if not project:
        raise RuntimeError(
            "Vertex AI project ID tidak ditemukan. "
            "Set VERTEX_PROJECT di .env atau pastikan SA JSON mengandung 'project_id'."
        )

    try:
        from google.oauth2 import service_account  # type: ignore

        credentials = service_account.Credentials.from_service_account_file(
            str(sa_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        genai.configure(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
        )
        logger.info(
            "Gemini SDK configured via vertex_ai",
            extra={"project": project, "location": location, "sa_file": sa_path.name},
        )
    except ImportError as exc:
        raise RuntimeError(
            "Paket 'google-auth' belum terpasang. Jalankan: pip install google-auth>=2.0.0"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Gagal konfigurasi Vertex AI: {exc}") from exc


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_gemini_model(
    model_name: str,
    system_instruction: Optional[str] = None,
) -> "genai.GenerativeModel":
    """
    Kembalikan GenerativeModel yang sudah dikonfigurasi.

    Akan memanggil configure_genai() secara otomatis jika belum pernah diinit.
    """
    configure_genai()
    kwargs: dict = {"model_name": model_name}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    return genai.GenerativeModel(**kwargs)


def get_langchain_chat_llm(model_name: str, temperature: float = 0.1, max_output_tokens: int = 512, **kwargs):
    """
    Kembalikan LangChain Chat LLM yang sesuai dengan mode aktif.

    - api_key   → ChatGoogleGenerativeAI
    - vertex_ai → ChatVertexAI
    """
    from app.core.config import get_settings
    settings = get_settings()
    mode = (settings.gemini_mode or "api_key").lower()

    if mode == "vertex_ai":
        try:
            from langchain_google_vertexai import ChatVertexAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Paket 'langchain-google-vertexai' belum terpasang. "
                "Jalankan: pip install langchain-google-vertexai>=1.0.0"
            ) from exc

        sa_path = _sa_json_path()
        project = settings.vertex_project
        location = settings.vertex_location or "us-central1"

        # Baca project dari SA JSON jika belum ada
        if not project and sa_path:
            try:
                with open(sa_path, "r", encoding="utf-8") as f:
                    sa_data = json.load(f)
                project = sa_data.get("project_id", "")
            except Exception:
                pass

        return ChatVertexAI(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            project=project,
            location=location,
            **kwargs,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )


def get_langchain_embeddings(model_name: str):
    """
    Kembalikan LangChain Embeddings yang sesuai dengan mode aktif.

    - api_key   → GoogleGenerativeAIEmbeddings
    - vertex_ai → VertexAIEmbeddings
    """
    from app.core.config import get_settings
    settings = get_settings()
    mode = (settings.gemini_mode or "api_key").lower()

    if mode == "vertex_ai":
        try:
            from langchain_google_vertexai import VertexAIEmbeddings  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Paket 'langchain-google-vertexai' belum terpasang. "
                "Jalankan: pip install langchain-google-vertexai>=1.0.0"
            ) from exc
        return VertexAIEmbeddings(model_name=model_name)
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.google_api_key,
        )


def reset_configuration() -> None:
    """Reset state konfigurasi (berguna untuk testing atau hot-reload mode)."""
    global _configured_mode
    with _lock:
        _configured_mode = None
    logger.info("Gemini SDK configuration reset")


def validate_service_account_json(content: bytes) -> dict:
    """
    Validasi isi file Service Account JSON.

    Args:
        content: Isi raw file JSON (bytes).

    Returns:
        dict SA data jika valid.

    Raises:
        ValueError: Jika file tidak valid.
    """
    if len(content) > 1 * 1024 * 1024:  # 1 MB
        raise ValueError("File SA JSON terlalu besar (maks 1 MB).")

    try:
        data = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"File bukan JSON yang valid: {exc}") from exc

    required_fields = ["type", "project_id", "private_key", "client_email"]
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        raise ValueError(
            f"SA JSON tidak lengkap, field berikut tidak ada: {', '.join(missing)}"
        )

    if data.get("type") != "service_account":
        raise ValueError(
            f"File bukan Service Account (type='{data.get('type')}', harus 'service_account')."
        )

    return data
