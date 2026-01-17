from pathlib import Path
from typing import Iterable, List, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore
from pydantic import BaseModel

from app.core.config import get_settings


class StoredVectorMetadata(BaseModel):
    source: str
    chunk_id: str


class VectorStoreManager:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._embedding_model: Embeddings | None = None
        self._vector_store: VectorStore | None = None

    @property
    def embedding_model(self) -> Embeddings:
        if self._embedding_model is None:
            if self.settings.use_bge:
                from langchain.embeddings import HuggingFaceEmbeddings

                self._embedding_model = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-large-en"
                )
            else:
                self._embedding_model = OpenAIEmbeddings(
                    model=self.settings.embedding_model,
                    openai_api_key=self.settings.openai_api_key,
                )
        return self._embedding_model

    @property
    def vector_store_path(self) -> Path:
        return self.settings.vector_dir / "faiss_index"

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            if self.vector_store_path.exists():
                self._vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
            else:
                self._vector_store = FAISS.from_texts(
                    texts=[],
                    embedding=self.embedding_model,
                )
        return self._vector_store

    def persist(self) -> None:
        if self._vector_store is not None:
            self.vector_store_path.mkdir(parents=True, exist_ok=True)
            self._vector_store.save_local(str(self.vector_store_path))

    def ingest_texts(
        self,
        texts: Iterable[str],
        source: str,
        chunk_size: int = 800,
        chunk_overlap: int = 80,
    ) -> int:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        documents: List[Document] = []
        for idx, raw_text in enumerate(texts):
            for chunk_id, chunk in enumerate(splitter.split_text(raw_text)):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source,
                            "chunk_id": f"{idx}-{chunk_id}",
                        },
                    )
                )
        if not documents:
            return 0
        self.vector_store.add_documents(documents)
        self.persist()
        return len(documents)

    def similarity_search_with_scores(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(
            query, k=top_k
        )


vector_store_manager = VectorStoreManager()

