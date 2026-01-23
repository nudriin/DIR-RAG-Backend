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


def rerank_documents(query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
    if not documents:
        return []

    try:
        from sentence_transformers import CrossEncoder

        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        cross_encoder = CrossEncoder(model_name)

        pairs = [(query, d.page_content) for d in documents]
        scores = cross_encoder.predict(pairs)

        scored = list(zip(documents, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        reranked = [doc for doc, _ in scored[: max(1, top_n)]]

        logger.info(
            "Reranked documents",
            extra={"query": query, "top_n": top_n, "num_input_docs": len(documents)},
        )
        return reranked
    except Exception as exc:
        logger.warning(
            "Reranker unavailable, using original order",
            extra={"error": str(exc), "top_n": top_n, "num_input_docs": len(documents)},
        )
        return documents[: max(1, top_n)]
