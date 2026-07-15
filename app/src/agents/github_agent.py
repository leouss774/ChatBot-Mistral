"""
Agent GitHub — cherche des repos, exemples de code, cookbooks liés à Mistral AI.

Contrat: def github_node(state: AgentState) -> dict
Retourne uniquement {"github_results": [str]}

Testable en standalone (bloc __main__ en bas), sans dépendre du graphe
LangGraph complet ni des autres agents de l'équipe.
"""
import os
import time
import requests
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

GITHUB_API_URL = "https://api.github.com/search/repositories"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # optionnel, augmente le rate limit
MISTRAL_ORG = "mistralai"
MAX_RESULTS = 5

_llm = None


def _get_llm() -> ChatMistralAI:
    """LLM léger utilisé uniquement pour extraire des mots-clés de recherche propres."""
    global _llm
    if _llm is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        _llm = ChatMistralAI(model="mistral-small-latest", temperature=0, api_key=api_key)
    return _llm


def _extract_keywords(question: str) -> str:
    """
    Transforme une question en langage naturel en une requête de recherche
    GitHub concise (2-5 mots-clés). Fallback sur la question brute si le
    LLM échoue (ne doit jamais bloquer l'agent).
    """
    try:
        llm = _get_llm()
        response = llm.invoke([
            ("system", "Extrait 2 à 5 mots-clés de recherche GitHub (technique, "
                       "en anglais si le terme technique est anglais) à partir de "
                       "la question suivante. Réponds UNIQUEMENT avec les mots-clés "
                       "séparés par des espaces, sans explication, sans ponctuation."),
            ("user", question),
        ])
        keywords = response.content.strip()
        return keywords if keywords else question
    except Exception:
        return question


def _search_github(query: str, org: str | None = None) -> list[dict]:
    """Appelle l'API de recherche de repos GitHub. Retourne une liste vide en cas d'échec."""
    q = f"{query} org:{org}" if org else query
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    try:
        resp = requests.get(
            GITHUB_API_URL,
            params={"q": q, "sort": "stars", "order": "desc", "per_page": MAX_RESULTS},
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 403:
            # Rate limit atteint (fréquent sans token)
            return []
        resp.raise_for_status()
        return resp.json().get("items", [])
    except requests.RequestException:
        return []


def _format_repo(repo: dict) -> str:
    name = repo.get("full_name", "unknown")
    desc = repo.get("description") or "Pas de description."
    stars = repo.get("stargazers_count", 0)
    url = repo.get("html_url", "")
    updated = repo.get("updated_at", "")[:10]
    return f"- **{name}** ⭐{stars} — {desc}\n  {url} (dernière MAJ: {updated})"


def github_node(state: dict) -> dict:
    """
    Node LangGraph pour l'agent GitHub.

    Args:
        state: AgentState avec state["messages"][-1].content = question

    Returns:
        dict avec la clé "github_results": [str] (une entrée = un résumé
        formaté de la recherche, ou un message d'erreur clair)
    """
    question = state["messages"][-1].content
    query = _extract_keywords(question)

    # 1. Priorité aux repos officiels de l'org mistralai
    repos = _search_github(query, org=MISTRAL_ORG)

    # 2. Fallback : recherche large incluant "mistral" si rien trouvé sur l'org
    source_label = "org:mistralai"
    if not repos:
        repos = _search_github(f"{query} mistral")
        source_label = "recherche large (mention mistral)"

    if not repos:
        return {
            "github_results": [
                f"Aucun repo GitHub trouvé pour la requête '{query}' "
                f"(possible rate limit GitHub sans token, ou requête trop spécifique)."
            ]
        }

    formatted = [_format_repo(r) for r in repos]
    header = f"Résultats GitHub ({source_label}) pour '{query}':"
    result_text = header + "\n" + "\n".join(formatted)

    return {"github_results": [result_text]}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    test_questions = [
        "Y a-t-il un cookbook GitHub pour faire du RAG avec Mistral ?",
        "Cherche des exemples d'intégration de Mistral avec LangChain",
        "Repos GitHub sur le fine-tuning de Mistral 7B",
        "SDK Python officiel de Mistral AI",
    ]

    for q in test_questions:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        fake_state = {"messages": [HumanMessage(content=q)]}
        result = github_node(fake_state)
        print(result["github_results"][0])
        time.sleep(1)  # petite pause pour ne pas se faire rate-limit pendant les tests
