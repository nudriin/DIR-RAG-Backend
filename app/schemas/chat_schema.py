from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="Pertanyaan mentah dari pengguna.")


class SourceInfo(BaseModel):
    id: int
    source: Optional[str]
    chunk_id: Optional[str]


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    iterations: int
    confidence: float


class IngestResponse(BaseModel):
    source: str
    num_chunks: int


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


class ChatResponseWithTrace(ChatResponse):
    trace: List[DebugTrace]

