from __future__ import annotations

from typing import Any

from app.fetchers import crawl_seed_pages, search_arxiv, search_github_repositories, utc_now_iso
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
        from app.src.agents.web_research import web_research_node
        res = web_research_node({"messages": [{"content": question}]})
        text_content = "\n\n".join(res.get("web_results", []))
        return [
            {
                "id": f"web_research::{hash(question)}",
                "text": text_content,
                "source_url": "https://arxiv.org",
                "title": f"Web Research for: {question}",
                "section": "Academic and Web Search Results",
                "source_type": "web_research",
                "agent": "web_research",
                "date_ingestion": utc_now_iso(),
            }
        ]

    if selected_agent == "github":
        from app.src.agents.github_agent import github_node
        res = github_node({"messages": [{"content": question}]})
        text_content = "\n\n".join(res.get("github_results", []))
        return [
            {
                "id": f"github_research::{hash(question)}",
                "text": text_content,
                "source_url": "https://github.com/mistralai",
                "title": f"GitHub Search for: {question}",
                "section": "MistralAI Repositories Results",
                "source_type": "github",
                "agent": "github",
                "date_ingestion": utc_now_iso(),
            }
        ]

    if selected_agent == "contacts":
        return crawl_seed_pages(["https://mistral.ai/"], source_type="contact", agent="contacts", max_pages=4)

    if selected_agent == "applications":
        docs = crawl_seed_pages(["https://docs.mistral.ai/", "https://mistral.ai/"], source_type="application", agent="applications", max_pages=6)
        github = []
        for query in GITHUB_SEARCH_QUERIES[:2]:
            github.extend(search_github_repositories(query))
        return docs + github

    return crawl_seed_pages(DOC_SEEDS, source_type="doc", agent="docs", max_pages=8)
