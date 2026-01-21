# query_refinement.py

from typing import List, TypedDict
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
    is_ambiguous: bool


AMBIGUOUS_TOKENS = {"ini", "itu", "tadi", "dia", "mereka", "jadwalnya"}


def is_ambiguous_query(query: str) -> bool:
    lowered = query.lower()
    if len(query.strip()) < 5:
        return True
    return any(token in lowered for token in AMBIGUOUS_TOKENS)


def _llm_rewrite_query(query: str) -> tuple[str, List[str]]:
    """
    LLM-based Query Rewriting sesuai konsep RQ-RAG:
    - rewrite query agar retrieval-friendly
    - decompose jika kompleks
    """
    prompt = f"""
Kamu adalah modul Query Rewriter dalam sistem Retrieval-Augmented Generation (RAG)
untuk domain pendidikan dan sistem akademik.

Tugasmu:
1. Perbaiki kueri agar lebih deskriptif dan eksplisit
2. Tambahkan konteks yang implisit jika perlu (kelas, modul, sistem akademik)
3. Jika kueri kompleks, pecah menjadi sub-kueri sederhana
4. Jangan menjawab pertanyaan pengguna

Kueri pengguna:
"{query}"

Keluarkan hasil dalam format JSON VALID berikut:
{{
  "refined_query": "...",
  "sub_queries": ["...", "..."]
}}
"""
    settings = get_settings()
    client = replicate.Client(api_token=settings.replicate_api_token)
    max_attempts = 3
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            output = client.run(
                settings.llm_model,
                input={
                    "prompt": prompt,
                    "temperature": 0.1,
                },
            )

            if isinstance(output, list):
                text = "".join(str(part) for part in output)
            else:
                text = str(output)

            # Ambil JSON dari output
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            parsed = json.loads(text[json_start:json_end])

            refined_query = parsed.get("refined_query", query)
            sub_queries = parsed.get("sub_queries", [refined_query])

            return refined_query.strip(), [q.strip() for q in sub_queries if q.strip()]

        except Exception as exc:
            last_exception = exc
            logger.error(
                "Query rewriting failed",
                extra={
                    "attempt": attempt,
                    "error": str(exc),
                    "llm_backend": "replicate",
                    "model": settings.llm_model,
                },
            )
            if attempt < max_attempts:
                time.sleep(3)

    # fallback jika LLM gagal total
    logger.warning("Fallback to rule-based query refinement")
    return query, [query]


def refine_query(query: str) -> RefinedQuery:
    query_clean = query.strip()
    ambiguous = is_ambiguous_query(query_clean)

    refined_query, sub_queries = _llm_rewrite_query(query_clean)

    # safety net
    if not sub_queries:
        sub_queries = [refined_query]

    result: RefinedQuery = {
        "original_query": query_clean,
        "refined_query": refined_query,
        "sub_queries": sub_queries,
        "is_ambiguous": ambiguous,
    }

    logger.info(
        "Refined query",
        extra={
            "original_query": query_clean,
            "refined_query": refined_query,
            "sub_queries": sub_queries,
            "is_ambiguous": ambiguous,
        },
    )

    return result
