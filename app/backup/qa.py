from __future__ import annotations

from app.agents import collect_live_chunks, detect_agent
from app.config import settings
from app.mistral_api import MistralClient
from app.rag_store import RAGStore


SYSTEM_PROMPT = """You are a focused assistant about Mistral AI.
Use only the retrieved context below.
If the context is insufficient, say what is missing.
Always cite sources as bullet points at the end.
Keep the answer concise, factual, and in French unless the user asks otherwise.
"""


class QAService:
    def __init__(self) -> None:
        self.store = RAGStore()
        self.llm = MistralClient() if settings.mistral_api_key else None

    def answer(self, question: str, agent: str | None = None, top_k: int | None = None) -> dict:
        selected_agent = agent or detect_agent(question)
        live_chunks = collect_live_chunks(question, selected_agent)
        if live_chunks and settings.mistral_api_key:
            try:
                self.store.upsert_chunks(live_chunks[: settings.max_live_chunks])
            except Exception:
                pass

        context_chunks = self.store.search(question, top_k=top_k or settings.max_context_chunks)
        context_text = self._format_context(context_chunks)

        if not self.llm:
            return {
                "agent": selected_agent,
                "answer": "MISTRAL_API_KEY manque. Le pipeline de récupération est prêt, mais la génération de réponse n'est pas encore activée.",
                "sources": context_chunks,
            }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\n\nContexte récupéré:\n{context_text}",
            },
        ]
        answer = self.llm.chat(messages)
        return {
            "agent": selected_agent,
            "answer": answer,
            "sources": context_chunks,
        }

    def ingest_urls(self, urls: list[str], source_type: str = "manual") -> int:
        from app.fetchers import crawl_seed_pages

        chunks = crawl_seed_pages(urls, source_type=source_type, agent="manual", max_pages=len(urls) * 2 or 1)
        if chunks and settings.mistral_api_key:
            self.store.upsert_chunks(chunks)
        return len(chunks)

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        parts = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.get("source_url", "")
            title = chunk.get("title") or ""
            parts.append(f"[{index}] {title}\nSource: {source}\n{chunk.get('text', '')}")
        return "\n\n".join(parts)
