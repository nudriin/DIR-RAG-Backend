from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="Pertanyaan mentah dari pengguna.")
    conversation_id: int | None = Field(
        default=None,
        description="ID percakapan sebelumnya (jika ingin melanjutkan).",
    )
    user_role: str | None = Field(
        default=None,
        description="Peran pengguna: siswa/pengajar/admin_sekolah/pengawas/dinas/umum.",
    )
    target_role: str | None = Field(
        default=None,
        description="Peran target jawaban jika berbeda dengan peran pengguna.",
    )


class SourceInfo(BaseModel):
    id: int
    source: Optional[str]
    chunk_id: Optional[str]


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    iterations: int
    confidence: float
    conversation_id: int


class IngestResponse(BaseModel):
    source: str
    num_chunks: int


class VectorResetResponse(BaseModel):
    success: bool


class VectorDeleteBySourceRequest(BaseModel):
    source: str


class VectorDeleteBySourceResponse(BaseModel):
    source: str
    deleted_count: int


class VectorSourceInfo(BaseModel):
    source: str
    num_chunks: int


class VectorSourceListResponse(BaseModel):
    sources: List[VectorSourceInfo]


class VectorSourceChunk(BaseModel):
    doc_id: str
    chunk_id: Optional[str]
    content: str


class VectorSourceDetailResponse(BaseModel):
    source: str
    num_chunks: int
    chunks: List[VectorSourceChunk]


class EvaluateRequest(BaseModel):
    questions: List[str]
    ground_truth_answers: Optional[List[str]] = None
    relevant_doc_ids: Optional[List[List[str]]] = None


class EvaluateResponse(BaseModel):
    hit_rate: float
    mrr: float
    ragas: Dict[str, float]


class DebugTrace(BaseModel):
    iteration: int
    refined_query: str
    num_documents: int
    retrieve: bool
    retrieval_confidence: float
    reason: str
    raw_query: str | None


class DebugDRAGIN(BaseModel):
    entropy: float
    confidence: float = 0.0
    should_retry: bool = False
    reason: str
    token_count: int = 0


class DebugRQDragin(BaseModel):
    iter_query: str
    current_draft: str
    new_docs_found: int
    pruning_discarded: int
    pruning_kept: int
    executing: str


class DebugETC(BaseModel):
    current_trend: str


class DebugIterationLog(BaseModel):
    step: int
    dragin: Optional[DebugDRAGIN] = None
    rq_dragin: Optional[DebugRQDragin] = None
    iter_retgen: Optional[DebugRQDragin] = None  # backward compatibility alias
    etc: DebugETC


class DebugRQ(BaseModel):
    refined_query: str
    sub_queries: List[str]
    docs_retrieved: int
    source_per_query: Dict[str, List[str]] = {}
    refinement_type: str = "REWRITE"


class DebugFinalStatus(BaseModel):
    stop_reason: str
    is_fallback: bool
    entropy_history: List[float]


class DebugLogs(BaseModel):
    rq_rag: DebugRQ
    iterations: List[DebugIterationLog]
    final_status: DebugFinalStatus


class ChatResponseWithTrace(ChatResponse):
    trace: List[DebugTrace]
    debug_logs: DebugLogs
    response_time_ms: Optional[float] = None
