import math
import numpy as np
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import retrieve_documents
from app.rag.generator import format_context

logger = get_logger(__name__)


@dataclass
class RetrievalDecision:
    retrieve: bool
    confidence: float
    reason: str
    entropy: float  # Tambahan untuk debugging riset Anda
    prompt: str | None


def calculate_entropy(top_logprobs: list) -> float:
    """
    Menghitung Shannon Entropy dari daftar logprobs.
    H = -sum(p * log(p))
    """
    # Ambil probabilitas asli (exp dari logprob)
    probs = [math.exp(lp.get("logprob", -100)) for lp in top_logprobs]

    # Normalisasi agar total = 1 (jika perlu)
    sum_probs = sum(probs)
    if sum_probs == 0:
        return 0
    probs = [p / sum_probs for p in probs]

    # Hitung entropy
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return entropy


def decide_retrieval_dragin(query: str) -> RetrievalDecision:
    settings = get_settings()
    documents = retrieve_documents(query)
    context_text = format_context(documents)
    # Inisialisasi LLM sesuai
    llm = ChatOpenAI(
        model=settings.gpt_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        logprobs=True,
        top_logprobs=settings.top_logprops,
    )

    # 1. Probing Stage: Minta LLM memberikan jawaban singkat (zero-shot)
    # Kita hanya butuh beberapa token awal untuk mengukur uncertainty
    decision_prompt = f"""Jawab singkat, apakah Anda memiliki informasi pasti mengenai: {query} Berikut adalah konteks dari dokumen yang relevan:\n\n"
        f"{context_text}\n\n"""
    response = llm.invoke(decision_prompt)

    # 2. Ekstrak Logprobs dari metadata LangChain
    # Struktur OpenAI API: response_metadata -> logprobs -> content -> [token_list]
    logprobs_content = response.response_metadata.get("logprobs", {}).get("content", [])

    if not logprobs_content:
        logger.warning("Logprobs tidak tersedia. Menggunakan fallback retrieval.")
        return RetrievalDecision(
            retrieve=True, confidence=0.0, reason="Logprobs missing", entropy=1.0
        )

    # 3. Hitung rata-rata entropy dari 3-5 token pertama (DRAGIN logic)
    token_entropies = []
    for token_data in logprobs_content[:5]:  # Mengambil 5 token pertama sebagai sampel
        top_lp = token_data.get("top_logprobs", [])
        token_entropies.append(calculate_entropy(top_lp))

    avg_entropy = np.mean(token_entropies)

    # 4. Decision Logic berdasarkan Threshold
    # Dalam DRAGIN, Entropy Tinggi = Uncertainty Tinggi = Harus Retrieve
    THRESHOLD = settings.dragin_threshold  # Rekomendasi: Mulai di angka 0.5 - 0.8

    retrieve = avg_entropy > THRESHOLD
    confidence = 1.0 - (avg_entropy / 2.5)  # Normalisasi kasar (max entropy ~2.5-3.0)

    if retrieve:
        reason = f"Entropy ({avg_entropy:.4f}) melampaui threshold. Model tidak yakin dengan data internal."
        last_prompt = None
    else:
        reason = f"Entropy ({avg_entropy:.4f}) rendah. Model yakin dapat menjawab tanpa retrieval tambahan."
        last_prompt = decision_prompt

    logger.info(
        "DRAGIN Retrieval Decision",
        extra={
            "query": query,
            "retrieve": retrieve,
            "entropy": avg_entropy,
            "confidence": confidence,
            "prompt": last_prompt,
        },
    )

    return RetrievalDecision(
        retrieve=retrieve,
        confidence=float(confidence),
        reason=reason,
        entropy=float(avg_entropy),
        prompt=last_prompt,
    )
