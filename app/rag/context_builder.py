import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logging import get_logger, broadcast_event

logger = get_logger(__name__)

CHARS_PER_TOKEN = 4
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?。\n])\s+')


@dataclass
class ScoredChunk:
    text: str
    doc_index: int
    chunk_index: int
    total_chunks: int
    score: float = 0.0
    token_estimate: int = 0
    metadata: dict = field(default_factory=dict)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_text_by_sentences(text: str) -> List[str]:
    sentences = _SENTENCE_BOUNDARY.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def split_into_chunks(
    documents: List[Document],
    chunk_size_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
) -> List[ScoredChunk]:
    settings = get_settings()
    size_tokens = chunk_size_tokens or settings.chunk_size_tokens
    overlap_tokens = chunk_overlap_tokens or settings.chunk_overlap_tokens
    size_chars = size_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    all_chunks: List[ScoredChunk] = []

    for doc_idx, doc in enumerate(documents):
        content = (doc.page_content or "").strip()
        if not content:
            continue
        sentences = _split_text_by_sentences(content)
        if not sentences:
            all_chunks.append(ScoredChunk(
                text=content, doc_index=doc_idx, chunk_index=0, total_chunks=1,
                token_estimate=_estimate_tokens(content), metadata=doc.metadata.copy(),
            ))
            continue

        doc_chunks: List[str] = []
        current_chunk_sentences: List[str] = []
        current_len = 0
        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len > size_chars and current_chunk_sentences:
                doc_chunks.append(" ".join(current_chunk_sentences))
                overlap_sentences: List[str] = []
                overlap_len = 0
                for prev_s in reversed(current_chunk_sentences):
                    if overlap_len + len(prev_s) > overlap_chars:
                        break
                    overlap_sentences.insert(0, prev_s)
                    overlap_len += len(prev_s)
                current_chunk_sentences = overlap_sentences
                current_len = overlap_len
            current_chunk_sentences.append(sentence)
            current_len += s_len
        if current_chunk_sentences:
            doc_chunks.append(" ".join(current_chunk_sentences))

        total = len(doc_chunks)
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            all_chunks.append(ScoredChunk(
                text=chunk_text, doc_index=doc_idx, chunk_index=chunk_idx, total_chunks=total,
                token_estimate=_estimate_tokens(chunk_text), metadata=doc.metadata.copy(),
            ))

    logger.info("Context chunking complete", extra={"num_documents": len(documents), "num_chunks": len(all_chunks)})
    return all_chunks


def _score_with_cross_encoder(query: str, chunks: List[ScoredChunk]) -> List[float]:
    import os, inspect
    import numpy as np
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    token = settings.hf_token
    if token:
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    ce_kwargs = {}
    sig = inspect.signature(CrossEncoder)
    if token and "token" in sig.parameters:
        ce_kwargs["token"] = token
    elif token and "use_auth_token" in sig.parameters:
        ce_kwargs["use_auth_token"] = token

    cross_encoder = CrossEncoder(settings.reranker_model, **ce_kwargs)
    pairs = [(query, chunk.text) for chunk in chunks]
    raw_scores = cross_encoder.predict(pairs)

    scores: List[float] = []
    for s in raw_scores:
        if hasattr(s, "shape"):
            arr = np.asarray(s)
            scores.append(float(arr.item()) if arr.size == 1 else float(arr.max()))
        else:
            scores.append(float(s))

    if scores:
        s_min, s_max = min(scores), max(scores)
        if s_max > s_min:
            scores = [(s - s_min) / (s_max - s_min) for s in scores]
        else:
            scores = [1.0] * len(scores)
    return scores


def _head_tail_fallback(chunks: List[ScoredChunk]) -> List[ScoredChunk]:
    if not chunks:
        return []
    sorted_chunks = sorted(chunks, key=lambda c: (c.doc_index, c.chunk_index))
    n = len(sorted_chunks)
    head_count = max(1, int(n * 0.7))
    tail_count = max(1, n - head_count)
    head = sorted_chunks[:head_count]
    tail = sorted_chunks[-tail_count:]
    seen = set()
    result: List[ScoredChunk] = []
    for c in head + tail:
        key = (c.doc_index, c.chunk_index)
        if key not in seen:
            seen.add(key)
            c.score = 1.0 if c in head else 0.5
            result.append(c)
    return result


def _role_boost(chunk_text: str, user_role: str | None) -> float:
    if not user_role:
        return 0.0
    role_keywords = {
        "pengajar": ["pengajar", "guru", "mengajar", "tenaga pendidik", "dashboard guru"],
        "siswa": ["siswa", "murid", "peserta didik", "pelajar", "dashboard siswa"],
        "admin_sekolah": ["admin sekolah", "operator", "tata usaha"],
        "pengawas": ["pengawas"],
        "dinas": ["dinas"],
    }
    keywords = role_keywords.get(user_role.strip().lower(), [user_role.strip().lower()])
    text_lower = chunk_text.lower()
    if any(k in text_lower for k in keywords):
        return 0.1
    return 0.0


def score_chunks(
    query: str, chunks: List[ScoredChunk],
    semantic_weight: float | None = None, positional_weight: float | None = None,
    user_role: str | None = None,
) -> List[ScoredChunk]:
    if not chunks:
        return []
    settings = get_settings()
    sw = semantic_weight if semantic_weight is not None else settings.scoring_semantic_weight
    pw = positional_weight if positional_weight is not None else settings.scoring_positional_weight

    try:
        semantic_scores = _score_with_cross_encoder(query, chunks)
        for i, chunk in enumerate(chunks):
            if chunk.total_chunks > 1:
                positional = 1.0 - (chunk.chunk_index / (chunk.total_chunks - 1))
            else:
                positional = 1.0
            chunk.score = sw * semantic_scores[i] + pw * positional + _role_boost(chunk.text, user_role)
        chunks.sort(key=lambda c: c.score, reverse=True)
        logger.info("Chunk scoring complete (hybrid)", extra={
            "num_chunks": len(chunks),
            "top_score": round(chunks[0].score, 4) if chunks else 0,
            "bottom_score": round(chunks[-1].score, 4) if chunks else 0,
        })
        broadcast_event(stage="context_builder", action="scoring",
            summary="Chunk scoring selesai (hybrid semantic + positional)",
            details={"num_chunks": len(chunks), "semantic_weight": sw, "positional_weight": pw})
    except Exception as exc:
        logger.warning(f"CrossEncoder scoring failed, using head+tail fallback: {exc}", extra={"error": str(exc)})
        broadcast_event(stage="context_builder", action="scoring_fallback",
            summary="Scoring gagal, menggunakan fallback head+tail", details={"error": str(exc)})
        chunks = _head_tail_fallback(chunks)
    return chunks


def select_top_chunks(scored_chunks: List[ScoredChunk], token_budget: int | None = None) -> List[ScoredChunk]:
    settings = get_settings()
    budget = token_budget or settings.context_token_budget
    selected: List[ScoredChunk] = []
    used_tokens = 0
    for chunk in scored_chunks:
        if used_tokens + chunk.token_estimate > budget:
            if not selected:
                selected.append(chunk)
                used_tokens += chunk.token_estimate
            break
        selected.append(chunk)
        used_tokens += chunk.token_estimate
    logger.info("Chunk selection complete", extra={
        "total_chunks": len(scored_chunks), "selected_chunks": len(selected),
        "used_tokens": used_tokens, "token_budget": budget,
    })
    broadcast_event(stage="context_builder", action="selection",
        summary=f"Terpilih {len(selected)} chunk dari {len(scored_chunks)} (budget {budget} token)",
        details={"selected": len(selected), "total": len(scored_chunks), "used_tokens": used_tokens, "budget": budget})
    return selected


def build_final_context(selected_chunks: List[ScoredChunk], preserve_order: bool = True) -> List[Document]:
    if not selected_chunks:
        return []
    if preserve_order:
        ordered = sorted(selected_chunks, key=lambda c: (c.doc_index, c.chunk_index))
    else:
        ordered = selected_chunks

    documents: List[Document] = []
    for chunk in ordered:
        meta = chunk.metadata.copy()
        meta["chunk_part"] = f"{chunk.chunk_index + 1}/{chunk.total_chunks}"
        meta["relevance_score"] = round(chunk.score, 4)
        documents.append(Document(page_content=chunk.text, metadata=meta))

    logger.info("Final context assembled", extra={"num_output_docs": len(documents), "preserve_order": preserve_order})
    broadcast_event(stage="context_builder", action="assembly",
        summary=f"Context final: {len(documents)} chunk, urutan {'asli' if preserve_order else 'by score'}",
        details={"num_chunks": len(documents)})
    return documents


def build_context_for_query(
    query: str, documents: List[Document],
    token_budget: int | None = None, preserve_order: bool = True,
    user_role: str | None = None,
) -> List[Document]:
    if not documents:
        return []
    chunks = split_into_chunks(documents)
    scored = score_chunks(query, chunks, user_role=user_role)
    selected = select_top_chunks(scored, token_budget)
    return build_final_context(selected, preserve_order)
