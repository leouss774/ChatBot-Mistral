"""
Agent Web Research — cherche des articles de recherche (Arxiv) et des
articles/actualités récentes sur le web à propos de Mistral AI.

Contrat: def web_research_node(state: AgentState) -> dict
Retourne uniquement {"web_results": [str]}

Testable en standalone (bloc __main__ en bas), sans dépendre du graphe
LangGraph complet ni des autres agents de l'équipe.

Stratégie recherche web (pas de setup requis par défaut) :
- Si TAVILY_API_KEY est présent dans .env -> utilise Tavily (meilleure qualité,
  pensé pour les agents LLM)
- Sinon -> fallback automatique sur duckduckgo-search (pip install
  duckduckgo-search), aucune clé API nécessaire, donc zéro setup pour démarrer
  vite en hackathon.
"""
import os
import arxiv
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
MAX_ARXIV_RESULTS = 4
MAX_WEB_RESULTS = 4


# ---------------------------------------------------------------------------
# Recherche Arxiv (papers de recherche)
# ---------------------------------------------------------------------------

def _search_arxiv(query: str) -> list[str]:
    """Cherche des papers Arxiv liés à la question, orienté Mistral AI/LLM."""
    try:
        search = arxiv.Search(
            query=f"{query} Mistral",
            max_results=MAX_ARXIV_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for r in search.results():
            authors = ", ".join(a.name for a in r.authors[:3])
            if len(r.authors) > 3:
                authors += " et al."
            summary = r.summary.replace("\n", " ")
            summary = summary[:280] + ("..." if len(summary) > 280 else "")
            results.append(
                f"- **{r.title}** ({r.published.strftime('%Y-%m-%d')}) — {authors}\n"
                f"  {summary}\n  {r.entry_id}"
            )
        return results
    except Exception as e:
        return [f"[Erreur recherche Arxiv: {e}]"]


# ---------------------------------------------------------------------------
# Recherche web générale (Tavily si dispo, sinon DuckDuckGo sans clé API)
# ---------------------------------------------------------------------------

def _search_web_tavily(query: str) -> list[str]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query=f"{query} Mistral AI", max_results=MAX_WEB_RESULTS)
    results = []
    for item in response.get("results", []):
        title = item.get("title", "Sans titre")
        url = item.get("url", "")
        content = (item.get("content") or "")[:250]
        results.append(f"- **{title}**\n  {content}...\n  {url}")
    return results


def _search_web_duckduckgo(query: str) -> list[str]:
    from duckduckgo_search import DDGS

    results = []
    with DDGS() as ddgs:
        hits = list(ddgs.text(f"{query} Mistral AI", max_results=MAX_WEB_RESULTS))
    for item in hits:
        title = item.get("title", "Sans titre")
        url = item.get("href", "")
        body = (item.get("body") or "")[:250]
        results.append(f"- **{title}**\n  {body}...\n  {url}")
    return results


def _search_web(query: str) -> list[str]:
    """Recherche web avec fallback automatique Tavily -> DuckDuckGo."""
    if TAVILY_API_KEY:
        try:
            return _search_web_tavily(query)
        except Exception as e:
            print(f"[WARN] Tavily a échoué ({e}), fallback sur DuckDuckGo.")

    try:
        return _search_web_duckduckgo(query)
    except Exception as e:
        return [f"[Erreur recherche web: {e}]"]


# ---------------------------------------------------------------------------
# Node LangGraph
# ---------------------------------------------------------------------------

def web_research_node(state: dict) -> dict:
    """
    Node LangGraph pour l'agent Web Research.

    Args:
        state: AgentState avec state["messages"][-1].content = question

    Returns:
        dict avec la clé "web_results": [str] — combine papers Arxiv et
        résultats web généraux, chacun préfixé pour rester lisible dans
        le synthesizer final.
    """
    question = state["messages"][-1].content

    arxiv_results = _search_arxiv(question)
    web_results = _search_web(question)

    combined = []
    if arxiv_results:
        combined.append("### Papers de recherche (Arxiv)\n" + "\n".join(arxiv_results))
    if web_results:
        combined.append("### Articles / actualités web\n" + "\n".join(web_results))

    if not combined:
        combined = ["Aucun résultat trouvé sur Arxiv ni sur le web pour cette question."]

    return {"web_results": ["\n\n".join(combined)]}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    test_questions = [
        "Quels sont les derniers papers de recherche sur les modèles Mixture of Experts de Mistral ?",
        "Actualités récentes sur les levées de fonds de Mistral AI",
        "Recherche sur les architectures state-space model comme Codestral Mamba",
        "Comparaison Mistral vs GPT en benchmarks récents",
    ]

    for q in test_questions:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        fake_state = {"messages": [HumanMessage(content=q)]}
        result = web_research_node(fake_state)
        print(result["web_results"][0])
