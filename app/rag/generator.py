from typing import Any, Dict, List, Tuple

import time
import replicate
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

    if not settings.replicate_api_token:
        logger.error("Missing REPLICATE_API_TOKEN")
        answer_text = (
            "Konfigurasi token Replicate belum diatur. "
            "Silakan set REPLICATE_API_TOKEN terlebih dahulu."
        )
    else:
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
                if isinstance(output, str):
                    answer_text = output
                elif isinstance(output, list):
                    answer_text = "".join(str(part) for part in output)
                else:
                    answer_text = str(output)
                break
            except Exception as exc:
                last_exception = exc
                message = str(exc)
                logger.error(
                    f"LLM generation failed on attempt {attempt}: {message}",
                    extra={
                        "error": message,
                        "llm_backend": "replicate",
                        "replicate_model": settings.llm_model,
                        "attempt": attempt,
                    },
                )
                if (
                    ("429" in message or "throttled" in message or "E001" in message)
                    and attempt < max_attempts
                ):
                    time.sleep(4)
                    continue
                break

        if last_exception is not None and "answer_text" not in locals():
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
