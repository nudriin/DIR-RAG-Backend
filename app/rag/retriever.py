from typing import List

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
