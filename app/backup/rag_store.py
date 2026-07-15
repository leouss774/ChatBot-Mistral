from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from app.config import settings
from app.mistral_api import MistralClient


class RAGStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_or_create_collection(name=settings.chroma_collection)
        self.llm = MistralClient() if settings.mistral_api_key else None

    def count(self) -> int:
        return self.collection.count()

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        if not self.llm:
            raise RuntimeError("Cannot index documents without MISTRAL_API_KEY")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.llm.embed_texts(texts)
        ids = [chunk.get("id") or f"chunk-{index}" for index, chunk in enumerate(chunks)]
        metadatas = [
            {
                "source_url": chunk.get("source_url", ""),
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "source_type": chunk.get("source_type", "web"),
                "agent": chunk.get("agent", "unknown"),
            }
            for chunk in chunks
        ]
        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.llm:
            return []
        query_embedding = self.llm.embed_texts([query])[0]
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            chunks.append(
                {
                    "text": document,
                    "source_url": metadata.get("source_url", ""),
                    "title": metadata.get("title"),
                    "section": metadata.get("section"),
                    "source_type": metadata.get("source_type", "web"),
                    "score": distances[index] if index < len(distances) else None,
                }
            )
        return chunks
