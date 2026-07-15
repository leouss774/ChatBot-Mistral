from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    agent: str | None = None
    top_k: int = 5


class IngestRequest(BaseModel):
    urls: list[str] = Field(default_factory=list)
    source_type: str = "manual"


class SourceChunk(BaseModel):
    text: str
    source_url: str
    title: str | None = None
    section: str | None = None
    source_type: str
    score: float | None = None


class QueryResponse(BaseModel):
    agent: str
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
