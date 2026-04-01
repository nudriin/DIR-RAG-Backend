from typing import Any, Dict, List, Tuple
import time
import replicate

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger, broadcast_event
from app.rag.context_builder import build_context_for_query

logger = get_logger(__name__)


def build_system_prompt() -> str:
    return (
        "Kamu adalah asisten AI resmi untuk sistem Kelas Digital Huma Betang (Humbet). "
        "Tugas utamamu adalah menjawab pertanyaan pengguna secara terstruktur, ringkas, "
        "dan akurat berdasarkan HANYA konteks dokumen resmi yang diberikan.\n\n"
        "Pedoman:\n"
        "1. Jawab langsung dan to-the-point, gunakan format poin jika informasi banyak.\n"
        "2. Jika informasi dalam konteks kurang lengkap, tetap berikan jawaban terbaik berdasarkan apa yang tersedia.\n"
        "3. JANGAN gunakan pengetahuan di luar dokumen konteks.\n"
        "4. Gunakan bahasa Indonesia baku dan formal."
    )


def format_context(docs: List[Document]) -> str:
    blocks: List[str] = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", f"doc-{i}")
        chunk_id = doc.metadata.get("chunk_id", "0")
        chunk_part = doc.metadata.get("chunk_part", "")
        relevance = doc.metadata.get("relevance_score", "")
        header = f"[Sumber: {source} | Chunk: {chunk_id}"
        if chunk_part:
            header += f" | Part: {chunk_part}"
        if relevance:
            header += f" | Skor: {relevance}"
        header += "]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def limit_docs_for_context(
    query: str,
    documents: List[Document],
    max_docs: int,
    max_chars: int,
) -> List[Document]:
    """
    Bangun context menggunakan relevance-aware semantic chunking.

    Fungsi ini sekarang mendelegasikan ke context_builder pipeline:
    chunk → score (CrossEncoder + positional) → select (token budget) → assemble.

    Args tetap sama untuk backward compatibility:
        query: Query pengguna.
        documents: List dokumen dari retrieval.
        max_docs: (legacy, tidak digunakan lagi — token budget menggantikan)
        max_chars: (legacy, tidak digunakan lagi — token budget menggantikan)

    Returns:
        List[Document] — chunk terpilih sebagai Document.
    """
    if not documents:
        return []
    return build_context_for_query(query=query, documents=documents)


def build_user_prompt(query: str, context_text: str) -> str:
    return (
        "Berikut adalah konteks dari dokumen yang relevan:\n\n"
        f"{context_text}\n\n"
        "Pertanyaan pengguna:\n"
        f"{query}\n\n"
        "Instruksi:\n"
        "- Jawab secara terstruktur, jelas, dan ringkas\n"
        "- Gunakan hanya informasi dari konteks\n"
        "- Jika informasi dalam konteks kurang lengkap, tetap berikan jawaban terbaik berdasarkan apa yang tersedia\n"
    )


def generate_answer(
    query: str, documents: List[Document]
) -> Tuple[str, List[Dict[str, Any]]]:

    settings = get_settings()

    docs_limited = limit_docs_for_context(
        query=query,
        documents=documents,
        max_docs=settings.context_max_docs,
        max_chars=settings.context_char_budget,
    )
    context_text = format_context(docs_limited)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(query=query, context_text=context_text)

    client = replicate.Client(api_token=settings.replicate_api_token)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    max_attempts = 3
    last_exception: Exception | None = None

    try:
        output = client.run(
            settings.llm_model,
            input={
                "prompt": full_prompt,
                "temperature": 0.1,
            },
        )
        text = (
            "".join(str(part) for part in output)
            if isinstance(output, list)
            else str(output)
        )
        answer_text = text

    except Exception as exc:
        last_exception = exc
        logger.error(
            f"LLM generation failed {str(exc)}",
            extra={
                "error": str(exc),
                "llm_backend": "replicate",
                "model": settings.llm_model,
            },
        )

    if last_exception is not None and "answer_text" not in locals():
        answer_text = (
            "Maaf, terjadi kesalahan teknis saat menghasilkan jawaban. "
            "Silakan coba kembali beberapa saat lagi."
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
            "model": settings.llm_model,
        },
    )
    broadcast_event(
        stage="generation",
        action="answer",
        summary="Jawaban akhir digenerate",
        details={"num_sources": len(sources)},
    )

    return answer_text, sources


def build_user_prompt_paragraph(
    query: str, context_text: str, previous_output: str | None
) -> str:
    continuation = ""
    if previous_output:
        continuation = (
            "Lanjutkan jawaban berikut dengan menambahkan satu paragraf baru yang relevan.\n\n"
            f"Jawaban sementara:\n{previous_output}\n\n"
        )
    return (
        "Berikut adalah konteks dari dokumen yang relevan:\n\n"
        f"{context_text}\n\n"
        "Pertanyaan pengguna:\n"
        f"{query}\n\n"
        f"{continuation}"
        "Instruksi:\n"
        "- Tulis hanya SATU paragraf baru (3–6 kalimat)\n"
        "- Jangan mengulang paragraf sebelumnya\n"
        "- Gunakan hanya informasi dari konteks\n"
        "- Jika informasi dalam konteks kurang lengkap, tetap tulis jawaban terbaik berdasarkan apa yang tersedia\n"
        "- Jangan gunakan pengetahuan internal di luar dokumen"
    )


def generate_paragraph(
    query: str,
    documents: List[Document],
    previous_output: str | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    settings = get_settings()

    docs_limited = limit_docs_for_context(
        query=query,
        documents=documents,
        max_docs=settings.context_max_docs,
        max_chars=settings.context_char_budget,
    )
    context_text = format_context(docs_limited)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt_paragraph(
        query=query, context_text=context_text, previous_output=previous_output
    )

    client = replicate.Client(api_token=settings.replicate_api_token)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    last_exception: Exception | None = None

    try:
        output = client.run(
            settings.llm_model,
            input={
                "prompt": full_prompt,
                "temperature": 0.1,
            },
        )
        text = (
            "".join(str(part) for part in output)
            if isinstance(output, list)
            else str(output)
        )
        paragraph_text = text
    except Exception as exc:
        last_exception = exc
        logger.error(
            f"LLM generation failed {str(exc)}",
            extra={
                "error": str(exc),
                "llm_backend": "replicate",
                "model": settings.llm_model,
            },
        )

    if last_exception is not None and "paragraph_text" not in locals():
        paragraph_text = "Tidak ada informasi tersedia untuk pertanyaan tersebut"

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
        "Generated paragraph",
        extra={
            "query": query,
            "num_sources": len(sources),
            "llm_backend": "replicate",
            "model": settings.llm_model,
        },
    )
    broadcast_event(
        stage="generation",
        action="paragraph",
        summary="Paragraf digenerate",
        details={"preview": paragraph_text[:120], "num_sources": len(sources)},
    )

    return paragraph_text, sources
