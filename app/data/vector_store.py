from pathlib import Path
from typing import Iterable, List, Tuple
import shutil

from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from langchain_experimental.text_splitter import SemanticChunker

from app.core.config import get_settings
from app.core.gemini_client import get_langchain_embeddings


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
                    model_name=self.settings.bge_model_name
                )
            else:
                self._embedding_model = get_langchain_embeddings(
                    model_name=self.settings.gemini_embedding_model,
                )
        return self._embedding_model

    @property
    def vector_store_path(self) -> Path:
        return self.settings.vector_dir / "chroma_db"

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = Chroma(
                persist_directory=str(self.vector_store_path),
                embedding_function=self.embedding_model,
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._vector_store

    def metadata_store(self) -> Chroma:
        return Chroma(
            persist_directory=str(self.vector_store_path),
        )

    def ingest_texts(
        self,
        texts: Iterable[str],
        source: str,
        chunk_size: int = 800,
        chunk_overlap: int = 80,
    ) -> int:
        splitter = SemanticChunker(
            self.embedding_model,
            breakpoint_threshold_type="percentile",
            # breakpoint_threshold_amount=self.settings.semantic_breakpoint_threshold_amount,
        )
        documents: List[Document] = []

        for idx, raw_text in enumerate(texts):
            semantic_docs = splitter.create_documents([raw_text])

            for chunk_id, doc in enumerate(semantic_docs):
                doc.metadata = {
                    "source": source,
                    "chunk_id": f"{idx}-{chunk_id}",
                }
                documents.append(doc)

        if not documents:
            return 0

        self.vector_store.add_documents(documents)
        return len(documents)

    def similarity_search_with_scores(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[Document, float]]:

        return self.vector_store.similarity_search_with_score(query, k=top_k)

    def reset_vectors(self) -> None:
        self._vector_store = None
        if self.vector_store_path.exists():
            shutil.rmtree(self.vector_store_path, ignore_errors=True)

    def delete_by_source(self, source: str) -> int:
        store = self.metadata_store()
        try:
            results = store.get(where={"source": source})
            ids_to_delete = results["ids"]

            if not ids_to_delete:
                return 0

            store.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception:
            return 0

    def list_sources(self) -> List[tuple[str, int]]:
        store = self.metadata_store()
        try:
            results = store.get(include=["metadatas"])
            metadatas = results["metadatas"]

            counts: dict[str, int] = {}
            for meta in metadatas:
                if meta:
                    src = meta.get("source")
                    if src:
                        counts[src] = counts.get(src, 0) + 1
            return sorted(counts.items(), key=lambda x: x[0].lower())
        except Exception:
            return []

    def get_documents_by_source(self, source: str) -> List[Tuple[str, Document]]:
        store = self.metadata_store()
        try:
            results = store.get(
                where={"source": source}, include=["metadatas", "documents"]
            )
            output: List[Tuple[str, Document]] = []

            ids = results["ids"]
            metadatas = results["metadatas"]
            docs = results["documents"]

            for i, doc_id in enumerate(ids):
                doc = Document(
                    page_content=docs[i], metadata=metadatas[i] if metadatas[i] else {}
                )
                output.append((doc_id, doc))
            return output
        except Exception:
            return []


vector_store_manager = VectorStoreManager()
