
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)


class SourceReference(BaseModel):
    source: str
    page_number: int
    chunk_id: int
    similarity_score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
    retrieved_chunks: int
