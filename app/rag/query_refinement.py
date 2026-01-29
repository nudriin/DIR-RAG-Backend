from typing import List, TypedDict, Optional
import json
import replicate
import time

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RefinedQuery(TypedDict):
    original_query: str
    refined_query: str
    sub_queries: List[str]
    refinement_type: str  # RQ-RAG: REWRITE, DECOMPOSE, atau DISAMBIGUATE


def refine_query(query: str, draft_answer: Optional[str] = None) -> RefinedQuery:
    """
    Refine query berdasarkan paradigma RQ-RAG dan ITER-RETGEN.
    Menggunakan draft_answer untuk 'bridge semantic gaps' jika tersedia.
    """
    settings = get_settings()

    # 1. KONSTRUKSI PROMPT BERDASARKAN RQ-RAG & ITER-RETGEN
    # Jika ada draft_answer, gunakan logika Generation-Augmented Retrieval [cite: 28]
    context_instruction = ""
    if draft_answer:
        context_instruction = f"""
        Berdasarkan draf jawaban sebelumnya: "{draft_answer}"
        Identifikasi celah informasi atau halusinasi yang terjadi. 
        Gunakan ini untuk merumuskan pencarian yang lebih akurat.
        """

    prompt = f"""
    Kamu adalah pakar optimasi kueri RAG untuk domain sistem informasi "Huma Betang".
    Tugasmu adalah melakukan Query Refinement berdasarkan kategori RQ-RAG:
    
    1. REWRITE: Jika kueri sederhana tapi kurang eksplisit.
    2. DECOMPOSE: Jika kueri kompleks/multi-hop, pecah menjadi langkah-langkah.
    3. DISAMBIGUATE: Jika kueri mengandung kata ambigu (ini, itu, jadwalnya).
    
    {context_instruction}
    
    Kueri Asli: "{query}"
    
    Berikan output dalam format JSON:
    {{
      "refinement_type": "REWRITE|DECOMPOSE|DISAMBIGUATE",
      "refined_query": "kueri tunggal yang paling optimal",
      "sub_queries": ["langkah 1", "langkah 2"]
    }}
    """

    # 2. EKSEKUSI VIA LLM
    client = replicate.Client(api_token=settings.replicate_api_token)

    try:
        output = client.run(
            settings.llm_model,
            input={
                "prompt": prompt,
                "temperature": 0.1,
            },
        )

        text = (
            "".join(str(part) for part in output)
            if isinstance(output, list)
            else str(output)
        )

        # Parsing JSON Safety
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        parsed = json.loads(text[json_start:json_end])

        result: RefinedQuery = {
            "original_query": query,
            "refined_query": parsed.get("refined_query", query),
            "sub_queries": parsed.get(
                "sub_queries", [parsed.get("refined_query", query)]
            ),
            "refinement_type": parsed.get("refinement_type", "REWRITE"),
        }

        logger.info(f"RQ-RAG Refinement Success: {result['refinement_type']}")
        return result

    except Exception as exc:
        logger.error(f"failed: {str(exc)}")

    # Fallback jika gagal
    return {
        "original_query": query,
        "refined_query": query,
        "sub_queries": [query],
        "refinement_type": "REWRITE",
    }
