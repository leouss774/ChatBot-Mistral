from fastapi import FastAPI, HTTPException

from app.qa import QAService
from app.schemas import IngestRequest, QueryRequest, QueryResponse, SourceChunk

app = FastAPI(title="Mistral Knowledge Bot", version="0.1.0")
qa_service = QAService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": qa_service.store.count()}


@app.post("/chat", response_model=QueryResponse)
def chat(payload: QueryRequest) -> QueryResponse:
    try:
        result = qa_service.answer(payload.question, agent=payload.agent, top_k=payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [SourceChunk(**source) for source in result["sources"]]
    return QueryResponse(agent=result["agent"], answer=result["answer"], sources=sources)


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict:
    try:
        count = qa_service.ingest_urls(payload.urls, source_type=payload.source_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ingested_chunks": count}
