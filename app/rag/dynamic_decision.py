import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.generator import build_system_prompt, format_context, limit_docs_for_context

import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import GenerationConfig as ProtoGenerationConfig

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
    Ekstrak logprobs dari response LANGSUNG google.generativeai SDK.

    Struktur proto LogprobsResult:
        logprobs_result.top_candidates[]     — list of TopCandidates (per posisi token)
        logprobs_result.chosen_candidates[]  — list of Candidate (token terpilih)

    top_candidates dan chosen_candidates adalah list PARALEL:
        top_candidates[i].candidates = [Candidate, ...] (top-k alternatif untuk token ke-i)
        chosen_candidates[i] = Candidate terpilih untuk token ke-i

    Kita normalisasi ke format [{top_logprobs: [{token, logprob}, ...]}, ...]
    agar calculate_token_entropy bisa dipakai langsung.
    """
    try:
        candidates = gemini_response.candidates
        if not candidates:
            logger.warning("Gemini response tidak punya candidates")
            return []

        logprobs_result = candidates[0].logprobs_result
        if not logprobs_result:
            logger.warning("Gemini candidate tidak punya logprobs_result")
            return []

        # top_candidates adalah list paralel di level logprobs_result
        top_candidates_list = getattr(logprobs_result, "top_candidates", None)

        if top_candidates_list:
            normalized = []
            for top_group in top_candidates_list:
                # Setiap top_group punya .candidates = list alternatif token
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
            logger.info(f"Gemini logprobs extracted: {len(normalized)} token entries via top_candidates")
            return normalized

        # Fallback: hanya chosen_candidates (tanpa alternatif, entropy=0)
        chosen = getattr(logprobs_result, "chosen_candidates", None)
        if chosen:
            logger.warning("Hanya chosen_candidates tersedia (tanpa top_candidates), entropy akan = 0")
            normalized = []
            for entry in chosen:
                top_logprobs = [
                    {
                        "token": getattr(entry, "token", ""),
                        "logprob": getattr(entry, "log_probability", -100),
                    }
                ]
                normalized.append({"top_logprobs": top_logprobs})
            return normalized

        logger.warning("logprobs_result tidak punya top_candidates maupun chosen_candidates")
        return []

    except Exception as exc:
        logger.warning(f"Gagal mengekstrak logprobs dari Gemini response: {exc}")
        return []


def _call_gemini_direct(
    system_prompt: str,
    user_prompt: str,
    settings,
) -> tuple:
    """
    Panggil Gemini API langsung via google.generativeai SDK.

    LangChain (langchain-google-genai) TIDAK meneruskan logprobs_result
    ke response_metadata, jadi kita harus bypass LangChain.

    Menggunakan proto GenerationConfig dari google.ai.generativelanguage_v1beta
    karena SDK wrapper (genai.types.GenerationConfig) tidak mendukung
    parameter response_logprobs dan logprobs.

    Returns:
        (answer_text, logprobs_content)
    """
    genai.configure(api_key=settings.google_api_key)

    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=system_prompt,
    )

    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": settings.max_generation_tokens,
        "response_logprobs": True,
        "logprobs": settings.top_logprops,
    }

    response = model.generate_content(
        user_prompt,
        generation_config=generation_config,
    )

    answer_text = response.text.strip()
    logprobs_content = _extract_logprobs_gemini(response)

    return answer_text, logprobs_content


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _create_llm(settings):
    """Buat LLM instance berdasarkan setting DRAGIN_LLM_BACKEND."""
    backend = settings.dragin_llm_backend.lower()

    if backend == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,
            max_output_tokens=settings.max_generation_tokens,
            top_logprobs=settings.top_logprops,
            response_logprobs=True,
        )
        logger.info(f"DRAGIN using Gemini backend: {settings.gemini_model}")
        return llm, "gemini"

    else:
        # Default: OpenAI
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.gpt_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
            logprobs=True,
            top_logprobs=settings.top_logprops,
            max_tokens=settings.max_generation_tokens,
        )
        logger.info(f"DRAGIN using OpenAI backend: {settings.gpt_model}")
        return llm, "openai"


# ---------------------------------------------------------------------------
# DRAGIN: unified generator + evaluator
# ---------------------------------------------------------------------------

def generate_with_dragin(
    query: str,
    documents: List[Document],
    sub_queries: List[str] | None = None,
    user_role: str | None = None,
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
        role_section = (
            "\nInformasi tentang pengguna:\n"
            f"- Peran pengguna saat ini: {user_role}\n"
            "- Jawab hanya untuk peran tersebut. Jika konteks menjelaskan "
            "fitur yang hanya tersedia untuk peran lain, jelaskan bahwa "
            "fitur tersebut tidak tersedia bagi peran pengguna dan jangan "
            "menyesatkan.\n"
        )

    user_prompt = (
        "Berikut adalah konteks dari dokumen yang relevan:\n\n"
        f"{context_text}\n\n"
        "Pertanyaan utama:\n"
        f"{query}\n"
        f"{sub_queries_section}"
        f"{role_section}\n"
        "Instruksi:\n"
        "- Jawab pertanyaan utama DAN semua sub-pertanyaan secara lengkap\n"
        "- Jawab secara terstruktur, jelas, dan ringkas\n"
        "- Gunakan hanya informasi dari konteks\n"
        "- Jika informasi dalam konteks kurang lengkap, tetap berikan jawaban terbaik berdasarkan apa yang tersedia\n"
    )

    # --- 2. Panggil LLM dengan logprobs ---
    backend = settings.dragin_llm_backend.lower()

    try:
        if backend == "gemini":
            # Bypass LangChain: langchain-google-genai tidak meneruskan logprobs
            logger.info(f"DRAGIN using Gemini direct SDK: {settings.gemini_model}")
            answer_text, logprobs_content = _call_gemini_direct(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                settings=settings,
            )
        else:
            # OpenAI via LangChain (logprobs didukung penuh)
            llm, backend = _create_llm(settings)
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            answer_text = response.content.strip()
            logprobs_content = _extract_logprobs_openai(response)
    except Exception as exc:
        logger.error(f"DRAGIN generation failed ({backend}): {exc}")
        return DRAGINResult(
            answer_text=(
                "Maaf, terjadi kesalahan teknis saat menghasilkan jawaban. "
                "Silakan coba kembali beberapa saat lagi."
            ),
            entropy=1.0,
            confidence=0.0,
            should_retry=False,
            reason=f"LLM error ({backend}): {exc}",
            token_count=0,
            llm_backend=backend,
        )

    if not logprobs_content:
        logger.warning(f"Logprobs tidak tersedia dari {backend}, anggap entropy tinggi")
        return DRAGINResult(
            answer_text=answer_text,
            entropy=1.0,
            confidence=0.0,
            should_retry=True,
            reason=f"Logprobs missing dari response ({backend})",
            token_count=0,
            llm_backend=backend,
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
            f"[{backend}] Entropy tinggi ({avg_entropy:.4f} > {THRESHOLD}), "
            f"confidence rendah ({confidence:.2f}). Perlu re-refine."
        )
    else:
        reason = (
            f"[{backend}] Entropy rendah ({avg_entropy:.4f} ≤ {THRESHOLD}), "
            f"confidence tinggi ({confidence:.2f}). Jawaban cukup."
        )

    logger.info(
        "DRAGIN evaluation",
        extra={
            "backend": backend,
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
        llm_backend=backend,
    )
