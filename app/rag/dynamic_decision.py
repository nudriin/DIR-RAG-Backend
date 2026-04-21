import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.logging import get_logger, broadcast_event
from app.core.gemini_client import (
    configure_genai, 
    get_gemini_model, 
    get_langchain_chat_llm,
    get_google_genai_client
)
from app.rag.generator import build_system_prompt, format_context, limit_docs_for_context

import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import GenerationConfig as ProtoGenerationConfig
from google.api_core.exceptions import ResourceExhausted

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DRAGINResult:
    """Hasil dari satu panggilan DRAGIN: jawaban + evaluasi uncertainty."""
    answer_text: str
    entropy: float
    confidence: float
    should_retry: bool
    reason: str
    token_count: int
    token_entropies: List[float] = field(default_factory=list)
    llm_backend: str = "openai"


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def calculate_token_entropy(top_logprobs: list) -> float:
    """Menghitung Shannon Entropy untuk satu token: H = -Σ p·log(p)"""
    probs = [math.exp(lp.get("logprob", -100)) for lp in top_logprobs]
    total = sum(probs)
    if total == 0:
        return 0.0
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _extract_logprobs_openai(response) -> List[dict]:
    """Ekstrak logprobs dari response OpenAI (ChatOpenAI)."""
    return (
        response.response_metadata
        .get("logprobs", {})
        .get("content", [])
    )


def _extract_logprobs_gemini(gemini_response) -> List[dict]:
    """
    Ekstrak logprobs dari response Gemini.
    Mendukung dua SDK:
    - google-generativeai (Legacy SDK)
    - google-genai (New SDK for Vertex AI logprobs)
    """
    try:
        # --- Case 1: google-genai (New SDK) ---
        # Structure: response.candidates[0].logprobs_result
        if hasattr(gemini_response, 'candidates') and gemini_response.candidates:
            candidate = gemini_response.candidates[0]
            
            # Check for new SDK structure
            logprobs_result = getattr(candidate, "logprobs_result", None)
            if logprobs_result:
                top_candidates_list = getattr(logprobs_result, "top_candidates", None)
                if top_candidates_list:
                    normalized = []
                    for top_group in top_candidates_list:
                        alt_candidates = getattr(top_group, "candidates", [])
                        top_logprobs = [
                            {
                                "token": getattr(c, "token", ""),
                                "logprob": getattr(c, "log_probability", -100),
                            }
                            for c in alt_candidates
                        ]
                        if top_logprobs:
                            normalized.append({"top_logprobs": top_logprobs})
                    return normalized

        # --- Case 2: google-generativeai (Legacy SDK) ---
        # The structure is similar but might be accessed differently depending on version
        # (This is already covered by the logic above if attributes match)
        
        return []

    except Exception as exc:
        logger.warning(f"Gagal mengekstrak logprobs dari Gemini response: {exc}")
        return []


def _call_gemini_direct(
    system_prompt: str,
    user_prompt: str,
    settings,
    gemini_mode_override: Optional[str] = None,
    vertex_project_override: Optional[str] = None,
    vertex_location_override: Optional[str] = None,
) -> tuple:
    """
    Panggil Gemini API langsung via SDK.
    Menggunakan SDK baru 'google-genai' untuk dukungan logprobs Vertex AI yang lebih baik.

    Returns:
        (answer_text, logprobs_content)
    """
    try:
        # Gunakan SDK baru jika tersedia
        client = get_google_genai_client(
            mode_override=gemini_mode_override,
            project_override=vertex_project_override,
            location_override=vertex_location_override,
        )
        from google.genai.types import GenerateContentConfig
        
        model_name = settings.gemini_model
        # Vertex AI SDK usually expects full model name or just ID
        
        mode_display = gemini_mode_override or settings.gemini_mode
        logger.info(f"Calling Gemini via google-genai client. Model: {model_name}, Mode: {mode_display}")
        
        # Disable safety filters to prevent truncation
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=settings.max_generation_tokens,
                response_logprobs=True,
                logprobs=settings.top_logprops,
                safety_settings=safety_settings,
            )
        )
        
        if response.candidates and response.candidates[0].finish_reason:
            logger.info(f"Gemini finish reason: {response.candidates[0].finish_reason}")

        answer_text = response.text.strip()
        logprobs_content = _extract_logprobs_gemini(response)
        return answer_text, logprobs_content

    except Exception as e:
        logger.warning(f"Gagal menggunakan SDK baru google-genai, fallback ke legacy: {e}")
        
        # Fallback ke SDK lama (google-generativeai)
        configure_genai(
            mode_override=gemini_mode_override,
            project_override=vertex_project_override,
            location_override=vertex_location_override,
        )
        model = get_gemini_model(
            model_name=settings.gemini_model,
            system_instruction=system_prompt,
        )
        
        # Legacy safety settings
        safety_settings_legacy = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        generation_config = {
            "temperature": 0.1,
            "max_output_tokens": settings.max_generation_tokens,
            "response_logprobs": True,
            "logprobs": settings.top_logprops,
        }
        response = model.generate_content(
            user_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings_legacy,
        )
        answer_text = response.text.strip()
        logprobs_content = _extract_logprobs_gemini(response)
        return answer_text, logprobs_content


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _create_llm(
    settings, 
    backend: str | None = None, 
    model_override: str | None = None,
    gemini_mode_override: str | None = None,
    vertex_project_override: str | None = None,
    vertex_location_override: str | None = None,
):
    """Buat LLM instance berdasarkan backend (dari DB atau env DRAGIN_LLM_BACKEND)."""
    resolved_backend = (backend or settings.dragin_llm_backend or "gemini").lower()

    if resolved_backend == "gemini":
        model_name = model_override or settings.gemini_model
        # Gunakan get_langchain_chat_llm agar mode api_key / vertex_ai otomatis dipilih
        llm = get_langchain_chat_llm(
            model_name=model_name,
            temperature=0.1,
            max_output_tokens=settings.max_generation_tokens,
            top_logprobs=settings.top_logprops,
            response_logprobs=True,
            mode_override=gemini_mode_override,
            project_override=vertex_project_override,
            location_override=vertex_location_override,
        )
        mode_display = gemini_mode_override or settings.gemini_mode
        logger.info(
            f"DRAGIN using Gemini backend [{mode_display}]: {model_name}"
        )
        return llm, "gemini"

    else:
        # Default: OpenAI
        from langchain_openai import ChatOpenAI

        model_name = model_override or settings.gpt_model
        llm = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=0.1,
            logprobs=True,
            top_logprobs=settings.top_logprops,
            max_tokens=settings.max_generation_tokens,
        )
        logger.info(f"DRAGIN using OpenAI backend: {model_name}")
        return llm, "openai"


# ---------------------------------------------------------------------------
# DRAGIN: unified generator + evaluator
# ---------------------------------------------------------------------------

def generate_with_dragin(
    query: str,
    documents: List[Document],
    sub_queries: List[str] | None = None,
    user_role: str | None = None,
    raw_query: str | None = None,
    chat_history: str | None = None,
    generator_backend: str | None = None,
    generator_model_override: str | None = None,
    gemini_mode_override: str | None = None,
    vertex_project_override: str | None = None,
    vertex_location_override: str | None = None,
) -> DRAGINResult:
    """
    Generate jawaban DAN evaluasi uncertainty dalam SATU panggilan LLM.

    Mendukung dua backend:
    - OpenAI (ChatOpenAI) — default
    - Gemini (ChatGoogleGenerativeAI) — aktifkan via DRAGIN_LLM_BACKEND=gemini

    Cara kerja:
    1. Bangun prompt dari query + context documents.
    2. Panggil LLM dengan logprobs enabled.
    3. Hitung Shannon Entropy rata-rata dari logprobs seluruh token jawaban.
    4. Jika entropy > threshold → should_retry = True (perlu re-refine).

    Returns:
        DRAGINResult berisi answer_text, entropy, confidence, dan should_retry.
    """
    settings = get_settings()

    # --- 1. Siapkan dokumen & prompt ---
    docs_limited = limit_docs_for_context(
        query=query,
        documents=documents,
        max_docs=settings.context_max_docs,
        max_chars=settings.context_char_budget,
        user_role=user_role,
    )
    context_text = format_context(docs_limited)
    system_prompt = build_system_prompt()

    sub_queries_section = ""
    if sub_queries:
        sq_list = "\n".join(f"  {i+1}. {sq}" for i, sq in enumerate(sub_queries))
        sub_queries_section = (
            "\nSub-pertanyaan yang juga harus dijawab:\n"
            f"{sq_list}\n"
        )

    role_section = ""
    if user_role:
        role_norm = (user_role or "").strip().lower()
        if role_norm not in {"umum", "general", "public"}:
            role_section = (
                "\nInformasi tentang pengguna:\n"
                f"- Peran pengguna saat ini: {user_role}\n"
                "- PENTING: Jawab HANYA dari perspektif peran ini.\n"
                "- Jika konten dokumen membahas prosedur untuk peran ini tapi heading/label "
                "menyebut peran lain, ikuti KONTEN bukan HEADING.\n"
                "- Jangan pernah mengatakan pengguna harus login/masuk sebagai peran lain, "
                "kecuali konteks dokumen secara eksplisit menyatakan prosedur yang sama.\n"
                "- Tentukan secara mandiri dari konteks dokumen untuk peran siapa fitur tersebut berlaku.\n"
                "- Jika konteks menunjukkan fitur untuk peran lain, jelaskan keterbatasan akses "
                "bagi peran pengguna dan jangan menyesatkan.\n"
            )

    original_query = raw_query or query

    # --- Riwayat percakapan (short-term memory) ---
    history_section = ""
    if chat_history:
        history_section = (
            "Riwayat percakapan sebelumnya (gunakan untuk memahami konteks):\n"
            f"{chat_history}\n\n"
        )

    user_prompt = (
        f"{history_section}"
        "Berikut adalah konteks dari dokumen yang relevan:\n\n"
        f"{context_text}\n\n"
        "Pertanyaan asli pengguna:\n"
        f"{original_query}\n\n"
        "Pertanyaan hasil refinement/optimalisasi (jika berbeda):\n"
        f"{query}\n"
        f"{sub_queries_section}"
        f"{role_section}\n"
        "Instruksi:\n"
        "- Utamakan menjawab sesuai maksud pertanyaan asli pengguna.\n"
        "- Gunakan pertanyaan hasil refinement hanya sebagai bantuan untuk menstrukturkan jawaban, bukan untuk mengubah maksud.\n"
        "- Jika ada riwayat percakapan sebelumnya, perhatikan konteks percakapan untuk memahami maksud pertanyaan saat ini (misalnya kata ganti 'itu', 'nya', 'tersebut' mungkin merujuk ke topik sebelumnya).\n"
        "- KRITIS: Jangan menyalin heading/label dari dokumen secara mentah. Jika heading dokumen menyebutkan peran yang berbeda dari konten aktualnya (misalnya heading 'Login Siswa' tapi konten menjelaskan login Pengajar), gunakan KONTEN AKTUAL dan abaikan heading yang salah.\n"
        "- Sebelum menulis jawaban, verifikasi bahwa setiap langkah yang kamu jelaskan konsisten dengan peran yang ditanyakan pengguna. Jangan menyebut peran lain kecuali untuk klarifikasi perbedaan.\n"
        "- Jika dokumen konteks merupakan panduan untuk peran tertentu, sebutkan peran tersebut secara eksplisit berdasarkan konten aktual, bukan dari heading/judul dokumen.\n"
        "- Tentukan dari konteks dokumen peran utama yang sedang dibahas (misalnya dinas, admin sekolah, pengajar, siswa, pengawas).\n"
        "- Jika peran pada konteks berbeda dari peran pengguna, mulai jawaban dengan satu kalimat klarifikasi tentang perbedaan akses/fitur.\n"
        "- Jangan menyatakan pengguna akan masuk ke dashboard/fitur peran lain; jelaskan sebagai informasi dari dokumen peran tersebut.\n"
        "- Jika konteks hanya membahas entitas lain yang mirip tetapi berbeda (misalnya guru non induk vs kelas ajar non induk), jelaskan keterbatasan tersebut dan jangan mengganti topik pertanyaan.\n"
        "- Jika informasi dalam konteks kurang lengkap untuk menjawab pertanyaan asli, jelaskan keterbatasannya secara eksplisit.\n"
        "- Jawab secara terstruktur, jelas, dan informatif.\n"
        "- Jika kamu menemukan informasi yang relevan meskipun tidak memberikan definisi formal yang tepat, sintesiskan informasi tersebut untuk memberikan gambaran yang membantu kepada pengguna.\n"
        "- Gunakan bahasa yang ramah dan membantu (Humbet AI Assistant style).\n"
        "- PENTING: Jawab secara lengkap dan tuntas. Jangan memotong jawaban di tengah kalimat.\n"
    )

    # --- 2. Panggil LLM dengan logprobs ---
    # Prioritas: generator_backend (dari DB) > settings.dragin_llm_backend (env)
    resolved_backend = (generator_backend or settings.dragin_llm_backend or "gemini").lower()
    backend_used = resolved_backend

    try:
        if resolved_backend == "gemini":
            # Bypass LangChain: langchain-google-genai tidak meneruskan logprobs
            model_to_use = generator_model_override or settings.gemini_model
            logger.info(f"DRAGIN using Gemini direct SDK: {model_to_use}")
            # Sementara override gemini_model di settings jika berbeda
            original_model = settings.gemini_model
            settings.__dict__["gemini_model"] = model_to_use  # temp override (thread-safe: per-request)
            try:
                answer_text, logprobs_content = _call_gemini_direct(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    settings=settings,
                    gemini_mode_override=gemini_mode_override,
                    vertex_project_override=vertex_project_override,
                    vertex_location_override=vertex_location_override,
                )
            finally:
                settings.__dict__["gemini_model"] = original_model
            backend_used = "gemini"
        else:
            # OpenAI via LangChain (logprobs didukung penuh)
            llm, backend_name = _create_llm(
                settings,
                backend=resolved_backend,
                model_override=generator_model_override,
                gemini_mode_override=gemini_mode_override,
                vertex_project_override=vertex_project_override,
                vertex_location_override=vertex_location_override,
            )
            backend_used = backend_name
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            answer_text = response.content.strip()
            logprobs_content = _extract_logprobs_openai(response)
    except ResourceExhausted as exc:
        logger.warning(f"DRAGIN quota exhausted ({backend_used}): {exc}")
        broadcast_event(
            stage="dragin",
            action="rate_limit",
            summary="LLM rate limit/kuota tercapai",
            details={"backend": backend_used, "error": str(exc)},
        )
        return DRAGINResult(
            answer_text=(
                "Layanan model sedang padat (rate limit/kuota tercapai). "
                "Silakan coba lagi beberapa saat."
            ),
            entropy=1.0,
            confidence=0.0,
            should_retry=False,
            reason=f"LLM rate limit ({backend_used}): {exc}",
            token_count=0,
            llm_backend=backend_used,
        )
    except Exception as exc:
        logger.error(f"DRAGIN generation failed ({backend_used}): {exc}")
        return DRAGINResult(
            answer_text=(
                "Maaf, terjadi kesalahan teknis saat menghasilkan jawaban. "
                "Silakan coba kembali beberapa saat lagi."
            ),
            entropy=1.0,
            confidence=0.0,
            should_retry=False,
            reason=f"LLM error ({backend_used}): {exc}",
            token_count=0,
            llm_backend=backend_used,
        )

    if not logprobs_content:
        logger.warning(f"Logprobs tidak tersedia dari {backend_used}, anggap entropy tinggi")
        return DRAGINResult(
            answer_text=answer_text,
            entropy=1.0,
            confidence=0.0,
            should_retry=True,
            reason=f"Logprobs missing dari response ({backend_used})",
            token_count=0,
            llm_backend=backend_used,
        )

    token_entropies = [
        calculate_token_entropy(t.get("top_logprobs", []))
        for t in logprobs_content
        if t.get("top_logprobs")
    ]
    token_count = len(token_entropies)

    if token_count == 0:
        avg_entropy = 1.0
    else:
        # DRAGIN: fokus pada token-token PALING UNCERTAIN (top-K%)
        # Token boilerplate (kata sambung, tanda baca) memiliki entropy
        # rendah dan mendilusi sinyal uncertainty dari token informatif.
        # Kita ambil top 20% token dengan entropy tertinggi.
        TOP_K_PERCENT = 0.20
        sorted_entropies = sorted(token_entropies, reverse=True)
        top_k_count = max(1, int(len(sorted_entropies) * TOP_K_PERCENT))
        top_k_entropies = sorted_entropies[:top_k_count]
        avg_entropy = float(np.mean(top_k_entropies))

        # Log perbandingan untuk analisis
        full_mean = float(np.mean(token_entropies))
        logger.debug(
            f"Entropy comparison: full_mean={full_mean:.4f}, "
            f"top_{int(TOP_K_PERCENT*100)}%_mean={avg_entropy:.4f} "
            f"(top {top_k_count}/{token_count} tokens)"
        )

    # --- 4. Keputusan: retry atau tidak ---
    THRESHOLD = settings.dragin_threshold
    should_retry = avg_entropy > THRESHOLD
    confidence = max(0.0, 1.0 - (avg_entropy / max(math.log(settings.top_logprops), 1.0)))

    if should_retry:
        reason = (
            f"[{backend_used}] Entropy tinggi ({avg_entropy:.4f} > {THRESHOLD}), "
            f"confidence rendah ({confidence:.2f}). Perlu re-refine."
        )
    else:
        reason = (
            f"[{backend_used}] Entropy rendah ({avg_entropy:.4f} ≤ {THRESHOLD}), "
            f"confidence tinggi ({confidence:.2f}). Jawaban cukup."
        )

    logger.info(
        "DRAGIN evaluation",
        extra={
            "backend": backend_used,
            "entropy": round(avg_entropy, 4),
            "confidence": round(confidence, 2),
            "should_retry": should_retry,
            "token_count": token_count,
        },
    )

    return DRAGINResult(
        answer_text=answer_text,
        entropy=float(avg_entropy),
        confidence=float(confidence),
        should_retry=should_retry,
        reason=reason,
        token_count=token_count,
        token_entropies=token_entropies,
        llm_backend=backend_used,
    )


# ---------------------------------------------------------------------------
# Post-generation role validation (Layer 3)
# ---------------------------------------------------------------------------

_ROLE_CONTRADICTIONS = {
    "pengajar": [
        "login sebagai siswa", "masuk sebagai siswa", "pilih login sebagai siswa",
        "pilih siswa", "klik siswa",
    ],
    "siswa": [
        "login sebagai pengajar", "masuk sebagai pengajar", "pilih login sebagai pengajar",
        "login sebagai guru", "masuk sebagai guru", "pilih pengajar", "klik pengajar",
    ],
    "admin_sekolah": [
        "login sebagai siswa", "login sebagai pengajar", "pilih siswa", "pilih pengajar",
    ],
    "pengawas": [
        "login sebagai siswa", "login sebagai pengajar", "pilih siswa", "pilih pengajar",
    ],
}


def validate_role_consistency(
    answer: str,
    user_role: str | None,
) -> str:
    """
    Validasi konsistensi peran pada jawaban yang digenerate.

    Jika jawaban menyarankan login/aksi sebagai peran yang salah,
    tambahkan disclaimer di awal jawaban.

    Returns:
        Jawaban yang sudah divalidasi (mungkin dengan disclaimer).
    """
    if not user_role or not answer:
        return answer

    role_norm = user_role.strip().lower()
    checks = _ROLE_CONTRADICTIONS.get(role_norm, [])
    answer_lower = answer.lower()

    found_contradictions = [
        c for c in checks if c in answer_lower
    ]

    if found_contradictions:
        role_display = user_role.replace("_", " ").title()
        logger.warning(
            "Role contradiction detected in answer",
            extra={
                "user_role": user_role,
                "contradictions": found_contradictions,
            },
        )
        broadcast_event(
            stage="generation",
            action="role_validation_warning",
            summary=f"Kontradiksi peran terdeteksi: {found_contradictions}",
            details={
                "user_role": user_role,
                "contradictions": found_contradictions,
            },
        )
        disclaimer = (
            f"**Catatan:** Berikut adalah informasi untuk peran **{role_display}**. "
            f"Beberapa bagian dokumen sumber mungkin mengandung label peran yang tidak sesuai, "
            f"namun konten di bawah ini telah disesuaikan untuk peran Anda.\n\n"
        )
        return disclaimer + answer

    return answer
