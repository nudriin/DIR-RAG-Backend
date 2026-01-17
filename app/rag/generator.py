from typing import Any, Dict, List, Tuple

from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.schema import HumanMessage, SystemMessage

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


def generate_answer(
    query: str, documents: List[Document]
) -> Tuple[str, List[Dict[str, Any]]]:
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.1,
        openai_api_key=settings.openai_api_key,
    )

    context_text = format_context(documents)

    messages = [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(
            content=(
                "Berikut konteks dari dokumen:\n"
                f"{context_text}\n\n"
                "Pertanyaan pengguna:\n"
                f"{query}\n\n"
                "Buat jawaban terstruktur, jelas, dan singkat. "
                "Di bagian akhir, tuliskan daftar sumber yang digunakan."
            )
        ),
    ]

    response = llm(messages)
    answer_text = response.content

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
        },
    )

    return answer_text, sources

