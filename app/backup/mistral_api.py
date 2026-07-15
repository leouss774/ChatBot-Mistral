from __future__ import annotations

from typing import Any

import requests

from app.config import settings


class MistralAPIError(RuntimeError):
    pass


class MistralClient:
    def __init__(self) -> None:
        if not settings.mistral_api_key:
            raise MistralAPIError("MISTRAL_API_KEY is not configured")
        self.base_url = settings.mistral_base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
        }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": settings.mistral_embed_model,
            "input": texts,
        }
        response = requests.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=self.headers,
            timeout=settings.request_timeout_s,
        )
        if response.status_code >= 400:
            raise MistralAPIError(f"Embedding request failed: {response.status_code} {response.text}")
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        payload = {
            "model": settings.mistral_chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=self.headers,
            timeout=settings.request_timeout_s,
        )
        if response.status_code >= 400:
            raise MistralAPIError(f"Chat request failed: {response.status_code} {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]
