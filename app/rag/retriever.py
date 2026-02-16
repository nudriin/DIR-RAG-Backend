from typing import List, Tuple

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger, broadcast_event
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
    broadcast_event(
        stage="retrieval",
        action="initial",
        summary="Retrieval dokumen awal",
        details={"query": query, "top_k": settings.similarity_top_k, "num_documents": len(documents)},
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
    broadcast_event(
        stage="retrieval",
        action="scored",
        summary="Retrieval dokumen dengan skor",
        details={"query": query, "top_k": k, "num_results": len(results)},
    )
    return results


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 5,
    min_score: float | None = None,
) -> List[Document]:
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
        import os
        import inspect
        import numpy as np
        from sentence_transformers import CrossEncoder

        primary_model = settings.reranker_model
        fallback_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
        token = settings.hf_token
        if token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

        ce_kwargs = {}
        sig = inspect.signature(CrossEncoder)
        if token and "token" in sig.parameters:
            ce_kwargs["token"] = token
        elif token and "use_auth_token" in sig.parameters:
            ce_kwargs["use_auth_token"] = token

        def build_cross_encoder(model_name: str) -> CrossEncoder:
            return CrossEncoder(model_name, **ce_kwargs)

        try:
            cross_encoder = build_cross_encoder(primary_model)
            model_used = primary_model
        except Exception as init_exc:
            if primary_model != fallback_model:
                cross_encoder = build_cross_encoder(fallback_model)
                model_used = fallback_model
            else:
                raise init_exc

        pairs = [(query, d.page_content) for d in unique_docs]
        raw_scores = cross_encoder.predict(pairs)
        scores: List[float] = []
        for s in raw_scores:
            if hasattr(s, "shape"):
                arr = np.asarray(s)
                if arr.size == 1:
                    scores.append(float(arr.item()))
                else:
                    scores.append(float(arr.max()))
            else:
                scores.append(float(s))

        scored = list(zip(unique_docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filter berdasarkan skor minimum
        reranked = [
            doc for doc, score in scored[:max(1, top_n)]
            if score >= min_score
        ]

        # Jika semua difilter, tetap kembalikan top-1
        if not reranked and scored:
            reranked = [scored[0][0]]

        logger.info(
            "Reranked documents (multilingual)",
            extra={
                "query": query,
                "model": model_used,
                "top_n": top_n,
                "num_input_docs": len(documents),
                "num_unique": len(unique_docs),
                "num_after_rerank": len(reranked),
                "min_score": min_score,
            },
        )
        broadcast_event(
            stage="reranker",
            action="complete",
            summary="Reranker selesai",
            details={
                "query": query,
                "model": model_used,
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
        broadcast_event(
            stage="reranker",
            action="fallback",
            summary="Reranker tidak tersedia, gunakan urutan awal",
            details={"error": str(exc), "top_n": top_n, "num_input_docs": len(documents)},
        )
        return unique_docs[:max(1, top_n)]
