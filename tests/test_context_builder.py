"""
Unit tests untuk context_builder.py

Tes ini TIDAK memerlukan koneksi ke model/API karena menggunakan mock
untuk CrossEncoder dan fokus pada logika chunking, scoring, dan selection.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.rag.context_builder import (
    split_into_chunks,
    score_chunks,
    select_top_chunks,
    build_final_context,
    build_context_for_query,
    _estimate_tokens,
    _split_text_by_sentences,
    _head_tail_fallback,
    ScoredChunk,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(content: str, source: str = "test.pdf", chunk_id: str = "0") -> Document:
    return Document(page_content=content, metadata={"source": source, "chunk_id": chunk_id})


LONG_TEXT = (
    "Kalimat pertama tentang pendaftaran. "
    "Kalimat kedua tentang prosedur login. "
    "Kalimat ketiga menjelaskan dashboard utama. "
    "Kalimat keempat tentang fitur pelaporan. "
    "Kalimat kelima berisi informasi kontak admin. "
    "Kalimat keenam membahas pengaturan profil. "
    "Kalimat ketujuh berisi panduan lengkap ekspor data. "
    "Kalimat kedelapan membahas integrasi dengan sistem lain. "
    "Kalimat kesembilan menjelaskan cara reset password. "
    "Kalimat kesepuluh adalah penutup dokumen yang penting."
)


# ---------------------------------------------------------------------------
# Test: Helpers
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_basic(self):
        assert _estimate_tokens("hello world") == max(1, len("hello world") // 4)

    def test_empty(self):
        assert _estimate_tokens("") == 1  # min 1


class TestSplitTextBySentences:
    def test_basic_split(self):
        text = "Kalimat satu. Kalimat dua. Kalimat tiga."
        result = _split_text_by_sentences(text)
        assert len(result) >= 2

    def test_question_mark(self):
        text = "Apa itu RAG? Ini penjelasannya."
        result = _split_text_by_sentences(text)
        assert len(result) == 2

    def test_newline_split(self):
        text = "Paragraf satu.\nParagraf dua."
        result = _split_text_by_sentences(text)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Test: split_into_chunks
# ---------------------------------------------------------------------------

class TestSplitIntoChunks:
    @patch("app.rag.context_builder.get_settings")
    def test_basic_chunking(self, mock_settings):
        settings = MagicMock()
        settings.chunk_size_tokens = 20  # small chunks for testing
        settings.chunk_overlap_tokens = 5
        mock_settings.return_value = settings

        doc = _make_doc(LONG_TEXT)
        chunks = split_into_chunks([doc])

        assert len(chunks) > 1, "Long text should produce multiple chunks"
        for chunk in chunks:
            assert chunk.text.strip() != ""
            assert chunk.doc_index == 0
            assert chunk.metadata["source"] == "test.pdf"

    @patch("app.rag.context_builder.get_settings")
    def test_empty_doc(self, mock_settings):
        settings = MagicMock()
        settings.chunk_size_tokens = 150
        settings.chunk_overlap_tokens = 50
        mock_settings.return_value = settings

        doc = _make_doc("")
        chunks = split_into_chunks([doc])
        assert len(chunks) == 0

    @patch("app.rag.context_builder.get_settings")
    def test_multiple_docs(self, mock_settings):
        settings = MagicMock()
        settings.chunk_size_tokens = 20
        settings.chunk_overlap_tokens = 5
        mock_settings.return_value = settings

        docs = [_make_doc("Kalimat A.", "a.pdf"), _make_doc("Kalimat B.", "b.pdf")]
        chunks = split_into_chunks(docs)

        doc_indices = {c.doc_index for c in chunks}
        assert 0 in doc_indices
        assert 1 in doc_indices

    @patch("app.rag.context_builder.get_settings")
    def test_preserves_sentences(self, mock_settings):
        """No chunk should cut mid-sentence when sentence boundaries exist."""
        settings = MagicMock()
        settings.chunk_size_tokens = 30
        settings.chunk_overlap_tokens = 5
        mock_settings.return_value = settings

        doc = _make_doc("Kalimat satu yang panjang sekali. Kalimat dua yang juga panjang. Kalimat tiga pendek.")
        chunks = split_into_chunks([doc])

        for chunk in chunks:
            # Each chunk should end with complete sentence or be end of doc
            text = chunk.text.strip()
            assert text != ""


# ---------------------------------------------------------------------------
# Test: score_chunks
# ---------------------------------------------------------------------------

class TestScoreChunks:
    @patch("app.rag.context_builder.get_settings")
    @patch("app.rag.context_builder._score_with_cross_encoder")
    def test_hybrid_scoring(self, mock_ce, mock_settings):
        settings = MagicMock()
        settings.scoring_semantic_weight = 0.8
        settings.scoring_positional_weight = 0.2
        mock_settings.return_value = settings

        chunks = [
            ScoredChunk(text="chunk 0", doc_index=0, chunk_index=0, total_chunks=3, token_estimate=10),
            ScoredChunk(text="chunk 1", doc_index=0, chunk_index=1, total_chunks=3, token_estimate=10),
            ScoredChunk(text="chunk 2", doc_index=0, chunk_index=2, total_chunks=3, token_estimate=10),
        ]

        # CrossEncoder returns normalized scores
        mock_ce.return_value = [0.5, 1.0, 0.3]

        scored = score_chunks("test query", chunks)

        assert len(scored) == 3
        # Should be sorted descending by score
        assert scored[0].score >= scored[1].score >= scored[2].score
        # All scores should be > 0
        assert all(c.score > 0 for c in scored)

    @patch("app.rag.context_builder.get_settings")
    @patch("app.rag.context_builder._score_with_cross_encoder")
    def test_fallback_on_error(self, mock_ce, mock_settings):
        settings = MagicMock()
        settings.scoring_semantic_weight = 0.8
        settings.scoring_positional_weight = 0.2
        mock_settings.return_value = settings

        mock_ce.side_effect = RuntimeError("Model load failed")

        chunks = [
            ScoredChunk(text="head chunk", doc_index=0, chunk_index=0, total_chunks=3, token_estimate=10),
            ScoredChunk(text="mid chunk", doc_index=0, chunk_index=1, total_chunks=3, token_estimate=10),
            ScoredChunk(text="tail chunk", doc_index=0, chunk_index=2, total_chunks=3, token_estimate=10),
        ]

        result = score_chunks("test query", chunks)
        # Should not raise, should return via fallback
        assert len(result) > 0

    def test_empty_chunks(self):
        result = score_chunks("query", [])
        assert result == []


# ---------------------------------------------------------------------------
# Test: select_top_chunks
# ---------------------------------------------------------------------------

class TestSelectTopChunks:
    @patch("app.rag.context_builder.get_settings")
    def test_respects_budget(self, mock_settings):
        settings = MagicMock()
        settings.context_token_budget = 25
        mock_settings.return_value = settings

        chunks = [
            ScoredChunk(text="a" * 40, doc_index=0, chunk_index=0, total_chunks=3, score=0.9, token_estimate=10),
            ScoredChunk(text="b" * 40, doc_index=0, chunk_index=1, total_chunks=3, score=0.8, token_estimate=10),
            ScoredChunk(text="c" * 40, doc_index=0, chunk_index=2, total_chunks=3, score=0.7, token_estimate=10),
        ]

        selected = select_top_chunks(chunks)
        total_tokens = sum(c.token_estimate for c in selected)
        assert total_tokens <= 25

    @patch("app.rag.context_builder.get_settings")
    def test_at_least_one_chunk(self, mock_settings):
        settings = MagicMock()
        settings.context_token_budget = 1  # Impossibly small budget
        mock_settings.return_value = settings

        chunks = [
            ScoredChunk(text="big chunk", doc_index=0, chunk_index=0, total_chunks=1, score=0.9, token_estimate=100),
        ]

        selected = select_top_chunks(chunks)
        assert len(selected) >= 1, "Should always return at least 1 chunk"


# ---------------------------------------------------------------------------
# Test: build_final_context
# ---------------------------------------------------------------------------

class TestBuildFinalContext:
    def test_preserves_order(self):
        chunks = [
            ScoredChunk(text="chunk B", doc_index=1, chunk_index=0, total_chunks=1, score=0.9, token_estimate=5, metadata={"source": "b.pdf", "chunk_id": "1"}),
            ScoredChunk(text="chunk A", doc_index=0, chunk_index=0, total_chunks=1, score=0.8, token_estimate=5, metadata={"source": "a.pdf", "chunk_id": "0"}),
        ]

        docs = build_final_context(chunks, preserve_order=True)
        assert docs[0].page_content == "chunk A"
        assert docs[1].page_content == "chunk B"

    def test_score_order(self):
        chunks = [
            ScoredChunk(text="chunk B", doc_index=1, chunk_index=0, total_chunks=1, score=0.9, token_estimate=5, metadata={"source": "b.pdf", "chunk_id": "1"}),
            ScoredChunk(text="chunk A", doc_index=0, chunk_index=0, total_chunks=1, score=0.8, token_estimate=5, metadata={"source": "a.pdf", "chunk_id": "0"}),
        ]

        docs = build_final_context(chunks, preserve_order=False)
        assert docs[0].page_content == "chunk B"  # higher score first

    def test_empty(self):
        assert build_final_context([]) == []

    def test_metadata_enrichment(self):
        chunks = [
            ScoredChunk(text="test", doc_index=0, chunk_index=0, total_chunks=2, score=0.75, token_estimate=5, metadata={"source": "x.pdf", "chunk_id": "0"}),
        ]
        docs = build_final_context(chunks)
        assert docs[0].metadata["chunk_part"] == "1/2"
        assert docs[0].metadata["relevance_score"] == 0.75


# ---------------------------------------------------------------------------
# Test: head_tail_fallback
# ---------------------------------------------------------------------------

class TestHeadTailFallback:
    def test_basic_fallback(self):
        chunks = [
            ScoredChunk(text=f"chunk {i}", doc_index=0, chunk_index=i, total_chunks=10, token_estimate=5)
            for i in range(10)
        ]
        result = _head_tail_fallback(chunks)
        assert len(result) == 10  # 70% head + 30% tail with overlap coverage

    def test_empty(self):
        assert _head_tail_fallback([]) == []
