from typing import Any, Dict, List, Tuple
import time

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import rerank_documents

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

def limit_docs_for_context(
    query: str,
    documents: List[Document],
    max_docs: int,
    max_chars: int,
) -> List[Document]:
    if not documents:
        return []
    top_docs = rerank_documents(query=query, documents=documents, top_n=max_docs)
    limited: List[Document] = []
    char_count = 0
    for d in top_docs:
        content_len = len(d.page_content or "")
        if char_count + content_len > max_chars:
            # Trim the last doc content to fit budget if useful
            remaining = max(0, max_chars - char_count)
            if remaining > 200:
                d_trim = Document(page_content=(d.page_content or "")[:remaining], metadata=d.metadata)
                limited.append(d_trim)
                break
            else:
                break
        limited.append(d)
        char_count += content_len
    return limited


def build_user_prompt(query: str, context_text: str) -> str:
    return (
        "Berikut adalah konteks dari dokumen yang relevan:\n\n"
        f"{context_text}\n\n"
        "Pertanyaan pengguna:\n"
        f"{query}\n\n"
        "Instruksi:\n"
        "- Jawab secara terstruktur, jelas, dan ringkas\n"
        "- Gunakan hanya informasi dari konteks\n"
        "- Jika jawaban tidak ditemukan di konteks, katakan bahwa kamu tidak tahu\n"
        "- Di bagian akhir, tuliskan daftar sumber yang digunakan"
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

    llm = ChatOpenAI(
        model=settings.gpt_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        max_tokens=settings.max_generation_tokens,
    )

    max_attempts = 3
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            answer_text = response.content
            break

        except Exception as exc:
            last_exception = exc
            logger.error(
                f"LLM generation failed {str(exc)}",
                extra={
                    "attempt": attempt,
                    "error": str(exc),
                    "llm_backend": "openai",
                    "model": settings.gpt_model,
                },
            )
            if attempt < max_attempts:
                time.sleep(3)

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
            "llm_backend": "openai",
            "model": settings.gpt_model,
        },
    )

    return answer_text, sources


def build_user_prompt_paragraph(query: str, context_text: str, previous_output: str | None) -> str:
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
        "- Jika informasi tidak ada di konteks, tulis: 'Tidak ada informasi tersedia untuk pertanyaan tersebut'\n"
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
    user_prompt = build_user_prompt_paragraph(query=query, context_text=context_text, previous_output=previous_output)

    llm = ChatOpenAI(
        model=settings.gpt_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        max_tokens=settings.max_generation_tokens,
    )

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    paragraph_text = response.content

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
            "llm_backend": "openai",
            "model": settings.gpt_model,
        },
    )

    return paragraph_text, sources
