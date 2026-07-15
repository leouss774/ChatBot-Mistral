from __future__ import annotations

from typing import Any

from app.fetchers import crawl_seed_pages, search_arxiv, search_github_repositories
from app.source_catalog import ARXIV_SEARCH_QUERIES, DOC_SEEDS, GITHUB_SEARCH_QUERIES


AGENT_KEYWORDS = {
    "docs": ["doc", "documentation", "api", "model", "models", "pricing", "changelog"],
    "web_research": ["paper", "benchmark", "research", "arxiv", "article"],
    "github": ["github", "repo", "repository", "code", "cookbook", "example"],
    "contacts": ["contact", "team", "sales", "support", "devrel", "linkedin"],
    "applications": ["application", "app", "use case", "demo", "starter", "template"],
}


def detect_agent(question: str) -> str:
    lower = question.lower()
    for agent, keywords in AGENT_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return agent
    return "docs"


def collect_live_chunks(question: str, agent: str | None = None) -> list[dict[str, Any]]:
    selected_agent = agent or detect_agent(question)

    if selected_agent == "web_research":
        chunks: list[dict[str, Any]] = []
        for query in ARXIV_SEARCH_QUERIES[:2]:
            chunks.extend(search_arxiv(query))
        return chunks

    if selected_agent == "github":
        chunks = []
        for query in GITHUB_SEARCH_QUERIES[:3]:
            chunks.extend(search_github_repositories(query))
        return chunks

    if selected_agent == "contacts":
        return crawl_seed_pages(["https://mistral.ai/"], source_type="contact", agent="contacts", max_pages=4)

    if selected_agent == "applications":
        docs = crawl_seed_pages(["https://docs.mistral.ai/", "https://mistral.ai/"], source_type="application", agent="applications", max_pages=6)
        github = []
        for query in GITHUB_SEARCH_QUERIES[:2]:
            github.extend(search_github_repositories(query))
        return docs + github

    return crawl_seed_pages(DOC_SEEDS, source_type="doc", agent="docs", max_pages=8)
