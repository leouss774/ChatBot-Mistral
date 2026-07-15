from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    mistral_api_key: str | None = os.getenv("MISTRAL_API_KEY")
    mistral_base_url: str = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    mistral_chat_model: str = os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest")
    mistral_embed_model: str = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
    chroma_path: Path = Path(os.getenv("CHROMA_PATH", "./data/chroma"))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "mistral_knowledge")
    github_token: str | None = os.getenv("GITHUB_TOKEN")
    request_timeout_s: int = int(os.getenv("REQUEST_TIMEOUT_S", "20"))
    max_context_chunks: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
    max_live_chunks: int = int(os.getenv("MAX_LIVE_CHUNKS", "12"))


settings = Settings()
