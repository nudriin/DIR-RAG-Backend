import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.generator import build_system_prompt, format_context, limit_docs_for_context

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


def _extract_logprobs_gemini(response) -> List[dict]:
    """
    Ekstrak logprobs dari response Gemini (ChatGoogleGenerativeAI).

    Gemini mengembalikan logprobs dalam format berbeda:
    response.generation_info atau response.response_metadata
    berisi 'logprobs_result' dengan 'chosen_candidates' dan 'top_candidates'.
    Kita normalisasi ke format yang sama dengan OpenAI agar
    calculate_token_entropy bisa dipakai langsung.
    """
    # Gemini via LangChain: logprobs ada di response_metadata
    metadata = response.response_metadata or {}

    # Path 1: langchain-google-genai format
    logprobs_result = metadata.get("logprobs_result")
    if logprobs_result:
        top_candidates = logprobs_result.get("top_candidates", [])
        normalized = []
        for candidate_group in top_candidates:
            candidates = candidate_group.get("candidates", [])
            top_logprobs = [
                {"token": c.get("token", ""), "logprob": c.get("log_probability", -100)}
                for c in candidates
            ]
            normalized.append({"top_logprobs": top_logprobs})
        return normalized

    # Path 2: google-genai SDK format (candidates.logprobs_result)
    candidates_meta = metadata.get("candidates", [])
    if candidates_meta:
        for cand in candidates_meta:
            lp_result = cand.get("logprobs_result")
            if not lp_result:
                continue
            top_candidates = lp_result.get("top_candidates", [])
            normalized = []
            for candidate_group in top_candidates:
                candidates = candidate_group.get("candidates", [])
                top_logprobs = [
                    {"token": c.get("token", ""), "logprob": c.get("log_probability", -100)}
                    for c in candidates
                ]
                normalized.append({"top_logprobs": top_logprobs})
            return normalized

    return []


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

    user_prompt = (
        "Berikut adalah konteks dari dokumen yang relevan:\n\n"
        f"{context_text}\n\n"
        "Pertanyaan pengguna:\n"
        f"{query}\n\n"
        "Instruksi:\n"
        "- Jawab secara terstruktur, jelas, dan ringkas\n"
        "- Gunakan hanya informasi dari konteks\n"
        "- Jika jawaban tidak ditemukan di konteks, katakan bahwa kamu tidak tahu\n"
        "- Di bagian akhir, tuliskan daftar sumber yang digunakan"
    )

    # --- 2. Panggil LLM dengan logprobs ---
    llm, backend = _create_llm(settings)

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
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

    answer_text = response.content.strip()

    # --- 3. Hitung Shannon Entropy dari logprobs jawaban ---
    if backend == "gemini":
        logprobs_content = _extract_logprobs_gemini(response)
    else:
        logprobs_content = _extract_logprobs_openai(response)

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
        avg_entropy = float(np.mean(token_entropies))

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
