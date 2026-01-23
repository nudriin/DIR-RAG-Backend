import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
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
    entropy: float
    prompt: Optional[str]

def calculate_entropy(top_logprobs: list) -> float:
    """Menghitung Shannon Entropy: H = -sum(p * log(p))"""
    probs = [math.exp(lp.get("logprob", -100)) for lp in top_logprobs]
    sum_probs = sum(probs)
    if sum_probs == 0:
        return 0
    probs = [p / sum_probs for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)

def decide_retrieval_dragin(query: str) -> RetrievalDecision:
    settings = get_settings()
    
    # 1. INITIAL RETRIEVAL (Ambil chunk awal untuk pengecekan)
    documents = retrieve_documents(query)
    context_text = format_context(documents)
    
    llm = ChatOpenAI(
        model=settings.gpt_model,
        api_key=settings.openai_api_key,
        temperature=0, # Suhu 0 untuk stabilitas probing
        logprobs=True,
        top_logprobs=5,
    )

    # 2. SUFFICIENCY PROBE (RQ-RAG & DRAGIN Hybrid Logic)
    # Memaksa model menjawab biner untuk mendapatkan probabilitas "Ya" vs "Tidak"
    decision_prompt = f"""
    Tugas: Evaluasi kecukupan informasi untuk domain "Huma Betang".
    Pertanyaan: {query}
    Konteks: {context_text}

    Apakah konteks di atas memuat informasi yang cukup untuk menjawab pertanyaan secara pasti dan detail?
    Jawab hanya dengan satu kata: Ya atau Tidak.
    """
    
    response = llm.invoke([HumanMessage(content=decision_prompt)])
    logprobs_content = response.response_metadata.get("logprobs", {}).get("content", [])

    if not logprobs_content:
        return RetrievalDecision(True, 0.0, "Logprobs missing", 1.0, None)

    # 3. ANALISIS LOGPROBS (TOKEN PERTAMA)
    first_token_data = logprobs_content[0].get("top_logprobs", [])
    
    # Cari probabilitas untuk jawaban "Ya"
    prob_ya = 0.0
    for lp in first_token_data:
        token_str = lp.get("token", "").lower().strip()
        if token_str in ["ya", "yes"]:
            prob_ya = math.exp(lp.get("logprob", -100))
            break

    # 4. HITUNG ENTROPY (DRAGIN Standard)
    token_entropies = [calculate_entropy(t.get("top_logprobs", [])) for t in logprobs_content[:3]]
    avg_entropy = np.mean(token_entropies)

    # 5. DECISION LOGIC
    # Informasi dianggap TIDAK CUKUP jika:
    # - Probabilitas "Ya" sangat rendah (Kebutuhan Retrieval Tinggi)
    # - ATAU Entropy tinggi (Ketidakpastian Model Tinggi)
    THRESHOLD_ENTROPY = settings.dragin_threshold # Default 0.5 - 0.8
    THRESHOLD_SUFFICIENCY = 0.7 # Jika model tidak 70% yakin informasinya ada, maka retrieve
    
    is_insufficient = prob_ya < THRESHOLD_SUFFICIENCY
    is_uncertain = avg_entropy > THRESHOLD_ENTROPY
    
    retrieve = is_insufficient or is_uncertain
    
    if retrieve:
        reason = f"Informasi kurang (Prob Ya: {prob_ya:.2f}) atau Entropy tinggi ({avg_entropy:.2f})."
        last_prompt = None # Picu RQ-RAG untuk dekomposisi
    else:
        reason = f"Informasi cukup (Prob Ya: {prob_ya:.2f}) dengan kepastian tinggi."
        last_prompt = decision_prompt

    return RetrievalDecision(
        retrieve=retrieve,
        confidence=float(prob_ya if not retrieve else 1.0 - prob_ya),
        reason=reason,
        entropy=float(avg_entropy),
        prompt=last_prompt,
    )
