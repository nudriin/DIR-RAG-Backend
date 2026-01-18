from typing import Any, Dict, List, Tuple
import os

from langchain_community.llms import Replicate
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)


def build_system_prompt() -> str:
    return (
        "Kamu adalah asisten AI untuk chatbot edukasi Humbet. "
        "Jawab hanya berdasarkan dokumen konteks yang diberikan. "
        "Jika informasi yang dibutuhkan tidak ada di konteks, katakan bahwa kamu tidak tahu. "
        "Selalu sebutkan sumber dokumen yang kamu gunakan."
    )


def format_context(docs: List[Document]) -> str:
    blocks: List[str] = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", f"doc-{i}")
        chunk_id = doc.metadata.get("chunk_id", "0")
        header = f"[Sumber: {source} | Chunk: {chunk_id}]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_prompt(query: str, context_text: str) -> str:
    return (
        f"{build_system_prompt()}\n\n"
        f"Berikut konteks dari dokumen:\n{context_text}\n\n"
        f"Pertanyaan pengguna:\n{query}\n\n"
        "Instruksi: Buat jawaban terstruktur, jelas, dan singkat dalam bahasa Indonesia. "
        "Jawab hanya berdasarkan konteks. Jika informasi tidak ada di konteks, katakan bahwa kamu tidak tahu. "
        "Di bagian akhir, tuliskan daftar sumber yang digunakan berdasarkan metadata sumber dan chunk."
    )


def generate_answer(
    query: str, documents: List[Document]
) -> Tuple[str, List[Dict[str, Any]]]:
    settings = get_settings()

    context_text = format_context(documents)
    prompt = build_prompt(query=query, context_text=context_text)

    if settings.replicate_api_token:
        os.environ["REPLICATE_API_TOKEN"] = settings.replicate_api_token

    llm = Replicate(
        model=settings.llm_model,
        model_kwargs={
            "temperature": 0.1,
        },
        replicate_api_token=settings.replicate_api_token,
    )

    try:
        answer_text = llm.invoke(prompt)
    except Exception as exc:
        logger.error(
            "LLM generation failed",
            extra={
                "error": str(exc),
                "llm_backend": "replicate",
                "replicate_model": settings.llm_model,
            },
        )
        answer_text = (
            "Maaf, terjadi kesalahan teknis saat menghubungi model bahasa. "
            "Silakan coba lagi beberapa saat lagi."
        )

    sources: List[Dict[str, Any]] = []
    for i, doc in enumerate(documents):
        sources.append(
            {
                "id": i,
                "source": doc.metadata.get("source"),
                "chunk_id": doc.metadata.get("chunk_id"),
            }
        )

    logger.info(
        "Generated answer",
        extra={
            "query": query,
            "num_sources": len(sources),
            "llm_backend": "replicate",
            "replicate_model": settings.llm_model,
        },
    )

    return answer_text, sources
