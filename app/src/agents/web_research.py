"""
src/agents/web_research.py
HAZEM — Web Research Agent + GitHub Agent
Branch: feature/research-github-hazem

Web Research Agent:
  - ArXiv: academic papers mentioning Mistral AI
  - Tavily: recent web articles/news (primary)
  - DuckDuckGo: fallback if no Tavily key
Combines both sources and returns {"web_results": [...]}.

Standalone-safe: uses a plain TypedDict for AgentState so it runs without
langgraph's native DLLs (which may be blocked by Application Control policies).
LangGraph is only required by Wiem's graph assembler (src/graph.py).
"""

import os
import time
from datetime import datetime, timezone
from typing import TypedDict, List, Any


# ── Lightweight AgentState (no langgraph dependency) ──────────────────────────
# In the full graph, Wiem imports AgentState from src.state directly.
# Here we define an equivalent plain TypedDict for standalone + unit-test use.
class AgentState(TypedDict, total=False):
    messages: List[Any]          # list of LangChain message objects
    next_agent: List[str]
    rag_results: List[str]
    web_results: List[str]
    github_results: List[str]
    contacts_results: List[str]
    applications_results: List[str]
    final_answer: str


# ── Minimal HumanMessage shim for standalone use ──────────────────────────────
class _HumanMessage:
    """Minimal shim so the standalone test block doesn't need langchain_core."""
    def __init__(self, content: str):
        self.content = content

try:
    from langchain_core.messages import HumanMessage  # used in real graph
except Exception:
    HumanMessage = _HumanMessage  # type: ignore[assignment,misc]


# ── ArXiv helper ───────────────────────────────────────────────────────────────

def _search_arxiv(query: str, max_results: int = 5) -> list[str]:
    """
    Search ArXiv for papers related to Mistral AI + the given query.
    Returns a list of formatted strings.
    """
    try:
        import arxiv  # pip install arxiv

        # arxiv 4.x API: use Client().results() — search.results() removed in 4.0
        client = arxiv.Client()
        search = arxiv.Search(
            query=f"Mistral AI {query}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            pub_date = paper.published.strftime("%Y-%m-%d") if paper.published else "N/A"
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            summary = paper.summary.replace("\n", " ")
            if len(summary) > 300:
                summary = summary[:297] + "..."
            results.append(
                f"📄 **{paper.title}**\n"
                f"   Authors: {authors} | Published: {pub_date}\n"
                f"   Summary: {summary}\n"
                f"   🔗 {paper.entry_id}"
            )
        return results

    except ImportError:
        return ["⚠️ `arxiv` package not installed. Run: pip install arxiv"]
    except Exception as e:
        return [f"⚠️ ArXiv search failed: {e}"]


# ── Tavily helper ──────────────────────────────────────────────────────────────

def _search_tavily(query: str, max_results: int = 5) -> list[str]:
    """
    Search the web using Tavily (LLM-optimized search API).
    Requires TAVILY_API_KEY env var.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []  # silently skip — caller will use DuckDuckGo fallback

    try:
        from tavily import TavilyClient  # pip install tavily-python

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=f"Mistral AI {query}",
            max_results=max_results,
            search_depth="advanced",
        )
        results = []
        for r in response.get("results", []):
            title   = r.get("title", "No title")
            url     = r.get("url", "")
            content = r.get("content", "")
            if len(content) > 300:
                content = content[:297] + "..."
            pub_date = r.get("published_date", "")
            date_str = f" | {pub_date}" if pub_date else ""
            results.append(
                f"🌐 **{title}**{date_str}\n"
                f"   {content}\n"
                f"   🔗 {url}"
            )
        return results

    except ImportError:
        return []  # will fall through to DuckDuckGo
    except Exception as e:
        return [f"⚠️ Tavily search failed: {e}"]


# ── DuckDuckGo fallback ────────────────────────────────────────────────────────

def _search_duckduckgo(query: str, max_results: int = 5) -> list[str]:
    """
    Fallback web search using DuckDuckGo (no API key required).
    Tries the new `ddgs` package first, falls back to `duckduckgo_search`.
    Install: pip install ddgs
    """
    # Import DDGS from duckduckgo_search library
    try:
        from duckduckgo_search import DDGS  # pip install duckduckgo-search
    except ImportError:
        return ["⚠️ Web search unavailable. Run: pip install duckduckgo-search"]

    try:
        results = []
        with DDGS() as ddgs_client:
            hits = ddgs_client.text(
                f"Mistral AI {query}",
                max_results=max_results,
            )
            for r in (hits or []):
                title = r.get("title", "No title")
                href  = r.get("href", r.get("url", ""))
                body  = r.get("body", r.get("snippet", ""))
                if len(body) > 300:
                    body = body[:297] + "..."
                results.append(
                    f"🌐 **{title}**\n"
                    f"   {body}\n"
                    f"   🔗 {href}"
                )
        return results

    except Exception as e:
        # DDG can be flaky — return a neutral message rather than crashing
        return [f"⚠️ DuckDuckGo search failed: {e}"]


# ── Agent node ─────────────────────────────────────────────────────────────────

def web_research_node(state: AgentState) -> dict:
    """
    Pure agent node — reads last message, returns {"web_results": [...]}.
    Combines ArXiv papers + Tavily (or DuckDuckGo fallback) web results.
    """
    msg = state["messages"][-1]
    if hasattr(msg, "content"):
        question = msg.content
    elif isinstance(msg, dict) and "content" in msg:
        question = msg["content"]
    else:
        question = str(msg)

    # 1️⃣ ArXiv academic papers
    arxiv_results = _search_arxiv(question, max_results=5)

    # 2️⃣ Web news — Tavily first, then DuckDuckGo
    web_hits = _search_tavily(question, max_results=5)
    source_label = "Tavily"
    if not web_hits:
        web_hits = _search_duckduckgo(question, max_results=5)
        source_label = "DuckDuckGo"

    # 3️⃣ Assemble output
    sections: list[str] = []

    if arxiv_results:
        sections.append(
            f"### 📚 Academic Papers (ArXiv — top {len(arxiv_results)})\n\n"
            + "\n\n".join(arxiv_results)
        )

    if web_hits:
        sections.append(
            f"### 🔎 Web Results ({source_label} — top {len(web_hits)})\n\n"
            + "\n\n".join(web_hits)
        )

    if not sections:
        return {
            "web_results": [
                "⚠️ No results found from ArXiv or web search for this query.\n"
                "Try installing: `pip install arxiv tavily-python` "
                "or set TAVILY_API_KEY."
            ]
        }

    header = f"## 🔬 Web Research Results for: \"{question}\"\n"
    footer = f"\n\n_Retrieved: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_"
    return {"web_results": [header + "\n\n".join(sections) + footer]}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "derniers papers sur les modèles Mixture of Experts",
        "function calling API",
        "Mistral Large context window benchmark",
        "fine-tuning open source LLM",
        "RAG retrieval augmented generation techniques",
    ]

    print("=" * 70)
    print("WEB RESEARCH AGENT — Standalone Test")
    print("=" * 70)

    for q in test_queries:
        print(f"\n🔍 Query: {q}")
        print("-" * 50)
        fake_state: AgentState = {
            "messages": [HumanMessage(content=q)],
            "next_agent": [],
            "rag_results": [],
            "web_results": [],
            "github_results": [],
            "contacts_results": [],
            "applications_results": [],
            "final_answer": "",
        }
        result = web_research_node(fake_state)
        for line in result["web_results"]:
            print(line)
        print()
        time.sleep(1)  # be polite with external APIs
