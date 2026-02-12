from typing import List, Tuple

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger
from app.data.vector_store import vector_store_manager


logger = get_logger(__name__)


def retrieve_documents(query: str) -> List[Document]:
    settings = get_settings()
    results = vector_store_manager.similarity_search_with_scores(
        query=query,
        top_k=settings.similarity_top_k,
    )
    documents = [doc for doc, _ in results]

    logger.info(
        "Retrieved documents",
        extra={
            "query": query,
            "top_k": settings.similarity_top_k,
            "num_documents": len(documents),
        },
    )

    return documents


def retrieve_documents_with_scores(
    query: str, top_k: int | None = None
) -> List[Tuple[Document, float]]:
    settings = get_settings()
    k = top_k if top_k is not None else settings.similarity_top_k
    results = vector_store_manager.similarity_search_with_scores(query=query, top_k=k)

    logger.info(
        "Retrieved documents with scores",
        extra={
            "query": query,
            "top_k": k,
            "num_results": len(results),
        },
    )
    return results


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 5,
    min_score: float | None = None,
) -> List[Document]:
    """Rerank documents menggunakan cross-encoder multilingual.

    Perbaikan dari versi lama:
    1. Model diganti ke multilingual (support Bahasa Indonesia).
    2. Deduplikasi berdasarkan konten sebelum reranking.
    3. Filter min_score untuk buang dokumen yang benar-benar tidak relevan.
    """
    if not documents:
        return []

    from app.core.config import get_settings
    settings = get_settings()
    if min_score is None:
        min_score = settings.reranker_min_score

    # --- Deduplikasi berdasarkan konten ---
    seen_hashes: set = set()
    unique_docs: List[Document] = []
    for doc in documents:
        content_hash = hash(doc.page_content)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    if not unique_docs:
        return []

    try:
        from sentence_transformers import CrossEncoder

        model_name = "cross-encoder/ms-marco-multilingual-MiniLM-L6-v2"
        cross_encoder = CrossEncoder(model_name)

        pairs = [(query, d.page_content) for d in unique_docs]
        scores = cross_encoder.predict(pairs)

        scored = list(zip(unique_docs, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        # Filter berdasarkan skor minimum
        reranked = [
            doc for doc, score in scored[:max(1, top_n)]
            if float(score) >= min_score
        ]

        # Jika semua difilter, tetap kembalikan top-1
        if not reranked and scored:
            reranked = [scored[0][0]]

        logger.info(
            "Reranked documents (multilingual)",
            extra={
                "query": query,
                "top_n": top_n,
                "num_input_docs": len(documents),
                "num_unique": len(unique_docs),
                "num_after_rerank": len(reranked),
                "min_score": min_score,
            },
        )
        return reranked
    except Exception as exc:
        logger.warning(
            "Reranker unavailable, using original order",
            extra={"error": str(exc), "top_n": top_n, "num_input_docs": len(documents)},
        )
        return unique_docs[:max(1, top_n)]
