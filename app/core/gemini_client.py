
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import google.generativeai as genai

# Disable AFC log dari google-genai SDK
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

if TYPE_CHECKING:
    from google import genai as new_genai_type

from app.core.logging import get_logger
from app.core.config import get_settings

logger = get_logger(__name__)


_lock = threading.Lock()
_configured_mode: Optional[str] = None  # track mode saat ini agar tidak re-init
_new_client: Optional["new_genai_type.Client"] = None


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


def configure_genai(
    force: bool = False,
    mode_override: Optional[str] = None,
    project_override: Optional[str] = None,
    location_override: Optional[str] = None,
) -> str:
    """
    Inisialisasi SDK Gemini sesuai GEMINI_MODE di settings atau override.

    Args:
        force: Paksa re-init meski sudah pernah dikonfigurasi.
        mode_override: "api_key" atau "vertex_ai".
        project_override: ID Proyek Vertex AI.
        location_override: Lokasi Vertex AI.

    Returns:
        Mode yang aktif: "api_key" atau "vertex_ai"

    Raises:
        RuntimeError: Jika kredensial tidak tersedia.
    """
    global _configured_mode

    settings = get_settings()
    mode = (mode_override or settings.gemini_mode or "api_key").lower()
    
    logger.debug(f"configure_genai called. Detected mode: {mode}. Current cached mode: {_configured_mode}")

    with _lock:
        if _configured_mode == mode and not force:
            return mode

        logger.info(f"Switching Gemini SDK mode to: {mode} (force={force})")
        if mode == "vertex_ai":
            _configure_vertex_ai(
                settings, 
                project_override=project_override, 
                location_override=location_override
            )
        else:
            _configure_api_key(settings)

        _configured_mode = mode
        return mode


def _configure_api_key(settings) -> None:
    """Konfigurasi Gemini via Google AI Studio API Key."""
    api_key = settings.google_api_key
    logger.debug(f"Configuring API Key mode. Key present: {bool(api_key)}")
    if not api_key:
        raise RuntimeError(
            "GEMINI_MODE=api_key tetapi GOOGLE_API_KEY tidak ditemukan di environment. "
            "Tambahkan GOOGLE_API_KEY ke file .env."
        )
    
    genai.configure(api_key=api_key)
    logger.info("Gemini SDK configured via api_key (Google AI Studio)")


def _configure_vertex_ai(
    settings, 
    project_override: Optional[str] = None, 
    location_override: Optional[str] = None
) -> None:
    """Konfigurasi Gemini via Vertex AI Service Account."""
    sa_path = _sa_json_path()
    logger.debug(f"Configuring Vertex AI mode. SA Path: {sa_path}")
    if sa_path is None:
        raise RuntimeError(
            "GEMINI_MODE=vertex_ai tetapi file Service Account JSON tidak ditemukan. "
            "Upload SA JSON melalui halaman Settings atau set GOOGLE_SERVICE_ACCOUNT_PATH di .env."
        )

    project = project_override or settings.vertex_project
    location = location_override or settings.vertex_location or "us-central1"
    logger.debug(f"Vertex AI initial config - project: {project}, location: {location}")

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
        
        # Konfigurasi Vertex AI menggunakan google-cloud-aiplatform
        import vertexai
        vertexai.init(
            project=project,
            location=location,
            credentials=credentials,
        )
        
        genai.configure()
        
        logger.info(
            "Gemini SDK configured via vertexai.init()",
            extra={"sa_file": sa_path.name, "project": project, "location": location},
        )
    except ImportError as exc:
        raise RuntimeError(
            "Paket 'google-cloud-aiplatform' atau 'google-auth' belum terpasang. "
            "Jalankan: pip install google-cloud-aiplatform google-auth>=2.0.0"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Gagal konfigurasi Vertex AI: {exc}") from exc


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_gemini_model(
    model_name: str,
    system_instruction: Optional[str] = None,
) -> Any:
    """
    Kembalikan GenerativeModel yang sudah dikonfigurasi.
    Mendukung API Key (google.generativeai) dan Vertex AI (vertexai).
    """
    settings = get_settings()
    mode = configure_genai()
    
    kwargs: dict = {"model_name": model_name}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
        
    if mode == "vertex_ai":
        from vertexai.generative_models import GenerativeModel
        logger.debug(f"Creating GenerativeModel via vertexai: {model_name}")
        return GenerativeModel(**kwargs)
    else:
        logger.debug(f"Creating GenerativeModel via google.generativeai: {model_name}")
        return genai.GenerativeModel(**kwargs)


def get_langchain_chat_llm(
    model_name: str, 
    temperature: float = 0.1, 
    max_output_tokens: int = 1024, 
    mode_override: Optional[str] = None,
    project_override: Optional[str] = None,
    location_override: Optional[str] = None,
    **kwargs
):
    """
    Kembalikan LangChain Chat LLM yang sesuai dengan mode aktif atau override.

    - api_key   → ChatGoogleGenerativeAI
    - vertex_ai → ChatVertexAI
    """
    from app.core.config import get_settings
    settings = get_settings()
    mode = (mode_override or settings.gemini_mode or "api_key").lower()
    
    logger.debug(f"get_langchain_chat_llm called for model: {model_name}. Detected mode: {mode}")

    if mode == "vertex_ai":
        try:
            from langchain_google_vertexai import ChatVertexAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Paket 'langchain-google-vertexai' belum terpasang. "
                "Jalankan: pip install langchain-google-vertexai>=1.0.0"
            ) from exc

        sa_path = _sa_json_path()
        project = project_override or settings.vertex_project
        location = location_override or settings.vertex_location or "us-central1"
        logger.debug(f"LangChain Vertex AI: project={project}, location={location}, SA={sa_path}")

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
        logger.debug(f"LangChain API Key mode for model: {model_name}")

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
    global _configured_mode, _new_client
    with _lock:
        _configured_mode = None
        _new_client = None
    logger.info("Gemini SDK configuration reset")


def get_google_genai_client(
    mode_override: Optional[str] = None,
    project_override: Optional[str] = None,
    location_override: Optional[str] = None,
) -> "new_genai_type.Client":
    """
    Inisialisasi dan kembalikan client dari SDK baru 'google-genai'.
    Mendukung logprobs pada Gemini 2.0+ via Vertex AI.
    """
    global _new_client, _configured_mode
    from google import genai as new_genai
    
    with _lock:
        settings = get_settings()
        mode = (mode_override or settings.gemini_mode or "api_key").lower()
        
        # Jika client sudah ada, pastikan mode-nya sama
        if _new_client:
            # Jika mode berbeda dari _configured_mode, kita reset.
            if _configured_mode and _configured_mode != mode:
                logger.info(f"Detected mode change from {_configured_mode} to {mode}. Resetting GenAI client.")
                _new_client = None
            else:
                return _new_client
            
        logger.debug(f"Initializing Google GenAI Client. Mode: {mode}")
        
        if mode == "vertex_ai":
            project = project_override or settings.vertex_project
            location = location_override or settings.vertex_location or "us-central1"
            sa_path = _sa_json_path()
            logger.debug(f"Google GenAI Vertex Config: project={project}, location={location}, SA={sa_path}")
            
            # Auto-read project from SA if missing
            if not project and sa_path:
                try:
                    with open(sa_path, "r", encoding="utf-8") as f:
                        sa_data = json.load(f)
                    project = sa_data.get("project_id", "")
                    logger.debug(f"Project ID inferred from SA: {project}")
                except Exception as e:
                    logger.warning(f"Failed to infer project from SA: {e}")
            
            if not project:
                raise RuntimeError("Vertex AI project ID is required for google-genai client.")
                
            client_config = {
                "vertexai": True,
                "project": project,
                "location": location,
            }
            
            # If using Service Account JSON, we need to pass credentials
            if sa_path:
                try:
                    from google.oauth2 import service_account
                    credentials = service_account.Credentials.from_service_account_file(
                        str(sa_path),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    client_config["credentials"] = credentials
                    logger.debug("Credentials loaded from SA JSON for Google GenAI Client")
                except ImportError:
                    logger.warning("google-auth not installed, skipping SA credentials for new client")
                    
            _new_client = new_genai.Client(**client_config)
            _configured_mode = mode # Update state
            logger.info(f"Google GenAI Client initialized via Vertex AI (project={project})")
        else:
            api_key = settings.google_api_key
            logger.debug(f"Google GenAI API Key mode. Key present: {bool(api_key)}")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY is required for google-genai client in api_key mode.")
            _new_client = new_genai.Client(api_key=api_key)
            _configured_mode = mode # Update state
            logger.info("Google GenAI Client initialized via API Key")
            
        return _new_client


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
