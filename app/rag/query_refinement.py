from typing import List, TypedDict

from app.core.logging import get_logger


logger = get_logger(__name__)


class RefinedQuery(TypedDict):
    original_query: str
    refined_query: str
    sub_queries: List[str]
    is_ambiguous: bool


def is_ambiguous_query(query: str) -> bool:
    lowered = query.lower()
    if len(query.strip()) < 5:
        return True
    ambiguous_tokens = ["ini", "itu", "tadi", "dia", "mereka", "jadwalnya"]
    if any(token in lowered for token in ambiguous_tokens):
        return True
    return False


def decompose_query(query: str) -> List[str]:
    if "dan" in query.lower():
        parts = [p.strip() for p in query.split("dan") if p.strip()]
        if len(parts) > 1:
            return parts
    return [query.strip()]


def refine_query(query: str) -> RefinedQuery:
    query_clean = query.strip()
    ambiguous = is_ambiguous_query(query_clean)

    if len(query_clean.split()) <= 3 and not query_clean.endswith("?"):
        refined = query_clean + " pada sistem akademik kampus?"
    else:
        refined = query_clean

    sub_queries = decompose_query(refined)

    result: RefinedQuery = {
        "original_query": query_clean,
        "refined_query": refined,
        "sub_queries": sub_queries,
        "is_ambiguous": ambiguous,
    }

    logger.info(
        "Refined query",
        extra={
            "original_query": query_clean,
            "refined_query": refined,
            "sub_queries": sub_queries,
            "is_ambiguous": ambiguous,
        },
    )

    return result

