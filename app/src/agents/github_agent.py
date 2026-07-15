"""
src/agents/github_agent.py
HAZEM — Web Research Agent + GitHub Agent
Branch: feature/research-github-hazem

GitHub Agent: searches mistralai org repos (and fallback to generic mistral query).
Handles rate-limits and auth errors gracefully — always returns something useful.

Standalone-safe: uses a plain TypedDict for AgentState so it runs without
langgraph's native DLLs (which may be blocked by Application Control policies).
LangGraph is only required by Wiem's graph assembler (src/graph.py).
"""

import os
import time
import requests
from typing import TypedDict, List, Any, Optional


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


# ── Constants ──────────────────────────────────────────────────────────────────
GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # optional — higher rate limit if set
TOP_N = 5
TIMEOUT = 10  # seconds

# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _search_repos(query: str, top_n: int = TOP_N) -> list[dict]:
    """
    Search GitHub repos. Tries org:mistralai first, then broader mistral query.
    Returns a list of formatted repo dicts, or [] on error.
    """
    headers = _build_headers()

    def _do_search(q: str) -> Optional[list]:
        url = f"{GITHUB_API_BASE}/search/repositories"
        params = {"q": q, "sort": "stars", "order": "desc", "per_page": top_n}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)

            # Rate-limit handling
            if resp.status_code == 403:
                reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
                wait = max(0, reset_ts - int(time.time()))
                return None, f"GitHub rate-limit hit. Resets in ~{wait}s."

            if resp.status_code == 422:
                return None, "GitHub rejected the query (invalid search terms)."

            resp.raise_for_status()
            items = resp.json().get("items", [])
            return items, None

        except requests.exceptions.Timeout:
            return None, "GitHub API timed out."
        except requests.exceptions.ConnectionError:
            return None, "Cannot reach GitHub API — check network."
        except Exception as e:
            return None, f"Unexpected error: {e}"

    # 1️⃣ Try org-scoped search first
    items, err = _do_search(f"{query} org:mistralai")
    if err:
        return [], err
    if items:
        return items, None

    # 2️⃣ Fallback: broader query
    items, err = _do_search(f"{query} mistral")
    if err:
        return [], err
    return items or [], None


def _format_repo(repo: dict) -> str:
    name = repo.get("full_name", "N/A")
    desc = repo.get("description") or "No description"
    url  = repo.get("html_url", "")
    stars = repo.get("stargazers_count", 0)
    lang  = repo.get("language") or "N/A"
    updated = (repo.get("updated_at") or "")[:10]  # YYYY-MM-DD
    return (
        f"📦 **{name}** ({stars:,} ⭐ | {lang} | updated {updated})\n"
        f"   {desc}\n"
        f"   🔗 {url}"
    )


# ── Agent node ─────────────────────────────────────────────────────────────────

def github_node(state: AgentState) -> dict:
    """
    Pure agent node — reads last message, returns {"github_results": [...]}.
    Compatible with LangGraph StateGraph.
    """
    msg = state["messages"][-1]
    if hasattr(msg, "content"):
        question = msg.content
    elif isinstance(msg, dict) and "content" in msg:
        question = msg["content"]
    else:
        question = str(msg)
    repos, err = _search_repos(question)

    if err and not repos:
        return {
            "github_results": [
                f"⚠️ GitHub search unavailable: {err}\n"
                "Tip: set the GITHUB_TOKEN env variable for higher rate limits.\n"
                "You can explore mistralai repos directly at: "
                "https://github.com/mistralai"
            ]
        }

    if not repos:
        return {
            "github_results": [
                "No GitHub repositories found for this query.\n"
                "Browse all mistralai repos: https://github.com/mistralai"
            ]
        }

    formatted = [_format_repo(r) for r in repos]
    header = f"## GitHub Repositories (top {len(formatted)} results for: \"{question}\")\n"
    footer = "\n\n_Source: GitHub Search API — mistralai org_"
    return {"github_results": [header + "\n\n".join(formatted) + footer]}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        "LLM fine-tuning examples",
        "function calling agents",
        "RAG retrieval augmented generation",
        "Mixtral mixture of experts",
        "mistral-inference deployment",
    ]

    print("=" * 70)
    print("GITHUB AGENT — Standalone Test")
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
        result = github_node(fake_state)
        for line in result["github_results"]:
            print(line)
        print()
