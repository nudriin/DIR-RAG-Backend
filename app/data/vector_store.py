from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
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
                from langchain_community.embeddings import HuggingFaceEmbeddings

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
    def vector_store(self) -> VectorStore | None:
        if self._vector_store is None and self.vector_store_path.exists():
            self._vector_store = FAISS.load_local(
                str(self.vector_store_path),
                self.embedding_model,
                allow_dangerous_deserialization=True,
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

        if self._vector_store is None:
            self._vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_model,
            )
        else:
            self._vector_store.add_documents(documents)

        self.persist()
        return len(documents)

    def similarity_search_with_scores(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search_with_score(query, k=top_k)

    def reset_vectors(self) -> None:
        self._vector_store = None
        if self.vector_store_path.exists():
            for path in self.vector_store_path.parent.glob("faiss_index*"):
                if path.is_file() or path.is_dir():
                    try:
                        if path.is_dir():
                            for child in path.glob("**/*"):
                                if child.is_file():
                                    child.unlink(missing_ok=True)
                            path.rmdir()
                        else:
                            path.unlink(missing_ok=True)
                    except FileNotFoundError:
                        continue

    def delete_by_source(self, source: str) -> int:
        store = self.vector_store
        if store is None:
            return 0
        try:
            docstore = store.docstore
        except AttributeError:
            return 0
        ids_to_delete: List[str] = []
        for doc_id, doc in getattr(docstore, "_dict", {}).items():
            if isinstance(doc, Document) and doc.metadata.get("source") == source:
                ids_to_delete.append(doc_id)
        if not ids_to_delete:
            return 0
        store.delete(ids_to_delete)
        self.persist()
        return len(ids_to_delete)

    def list_sources(self) -> List[tuple[str, int]]:
        store = self.vector_store
        if store is None:
            return []
        try:
            docstore = store.docstore
        except AttributeError:
            return []
        counts: dict[str, int] = {}
        for doc in getattr(docstore, "_dict", {}).values():
            if isinstance(doc, Document):
                src = doc.metadata.get("source")
                if not src:
                    continue
                counts[src] = counts.get(src, 0) + 1
        return sorted(counts.items(), key=lambda x: x[0].lower())


vector_store_manager = VectorStoreManager()
