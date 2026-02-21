from pathlib import Path
from typing import Iterable, List, Tuple
import shutil

from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from langchain_experimental.text_splitter import SemanticChunker

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
                    model_name=self.settings.bge_model_name
                )
            else:
                self._embedding_model = GoogleGenerativeAIEmbeddings(
                    model=self.settings.gemini_embedding_model,
                    google_api_key=self.settings.google_api_key,
                )
        return self._embedding_model

    @property
    def vector_store_path(self) -> Path:
        # ChromaDB persistence directory
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
            breakpoint_threshold_amount=self.settings.semantic_breakpoint_threshold_amount,
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

        # Simpan ke ChromaDB
        self.vector_store.add_documents(documents)
        return len(documents)

    def similarity_search_with_scores(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        # Chroma returns distance by default for cosine space (lower is better)
        # We might want to convert to similarity score if needed, but standard RAG often uses distance directly or converts 1-distance.
        # LangChain's similarity_search_with_score returns the raw score from the backend.
        # For Chroma with "cosine", it returns cosine distance (0 to 2). 0 means identical.
        return self.vector_store.similarity_search_with_score(query, k=top_k)

    def reset_vectors(self) -> None:
        self._vector_store = None
        if self.vector_store_path.exists():
            # Delete the entire directory for Chroma
            shutil.rmtree(self.vector_store_path, ignore_errors=True)

    def delete_by_source(self, source: str) -> int:
        store = self.metadata_store()
        # We need to find how many documents to delete first to return the count,
        # or just delete and return unknown count?
        # The user interface expects a count.

        # Get ids to delete first
        try:
            # Chroma specific method to get data
            # Note: store is a langchain wrapper around chroma client
            # Accessing internal client might be needed for complex queries or use get()

            # Using get() with filter
            results = store.get(where={"source": source})
            ids_to_delete = results["ids"]

            if not ids_to_delete:
                return 0

            store.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception:
            # Fallback if get/delete fails
            return 0

    def list_sources(self) -> List[tuple[str, int]]:
        store = self.metadata_store()
        try:
            # Get all metadata
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
                # Reconstruct Document
                doc = Document(
                    page_content=docs[i], metadata=metadatas[i] if metadatas[i] else {}
                )
                output.append((doc_id, doc))
            return output
        except Exception:
            return []


vector_store_manager = VectorStoreManager()
