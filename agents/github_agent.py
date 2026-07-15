"""
github_agent.py — Agent GitHub

Interroge l'API GitHub publique en direct (pas besoin d'auth pour les
endpoints publics en usage modéré) pour répondre à des questions nécessitant
des informations à jour : dernière release, nombre d'étoiles, description
du dépôt, etc. Contrairement au RAG, ces données ne sont jamais périmées.

Usage :
    python agents/github_agent.py "Quelle est la dernière release de mistral-inference ?"
"""

import os
import re
import sys
import httpx
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

GITHUB_API_BASE = "https://api.github.com"
CHAT_MODEL = "mistral-small-latest"

# Dépôts connus du projet (garde en phase avec ce qui est scrapé dans data/raw/github/)
KNOWN_REPOS = {
    "client-python": "mistralai/client-python",
    "mistral-inference": "mistralai/mistral-inference",
    "cookbook": "mistralai/cookbook",
}

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # optionnel, augmente juste la limite de rate

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def _headers():
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def identify_repo(question: str) -> str | None:
    """Trouve quel dépôt connu est mentionné dans la question (recherche simple par mot-clé)."""
    question_lower = question.lower()
    for keyword, repo in KNOWN_REPOS.items():
        if keyword.replace("-", " ") in question_lower.replace("-", " ") or keyword in question_lower:
            return repo
    # repli : cherche un motif "owner/repo" explicite dans la question
    match = re.search(r"[\w.-]+/[\w.-]+", question)
    if match:
        return match.group(0)
    return None


def fetch_repo_info(repo: str) -> dict:
    with httpx.Client(headers=_headers(), timeout=15) as http:
        repo_resp = http.get(f"{GITHUB_API_BASE}/repos/{repo}")
        repo_resp.raise_for_status()
        repo_data = repo_resp.json()

        releases_resp = http.get(f"{GITHUB_API_BASE}/repos/{repo}/releases/latest")
        latest_release = releases_resp.json() if releases_resp.status_code == 200 else None

    return {
        "full_name": repo_data.get("full_name"),
        "description": repo_data.get("description"),
        "stars": repo_data.get("stargazers_count"),
        "forks": repo_data.get("forks_count"),
        "open_issues": repo_data.get("open_issues_count"),
        "url": repo_data.get("html_url"),
        "latest_release_tag": latest_release.get("tag_name") if latest_release else None,
        "latest_release_date": latest_release.get("published_at") if latest_release else None,
        "latest_release_notes": (latest_release.get("body") or "")[:1000] if latest_release else None,
    }


def answer_question(question: str) -> dict:
    repo = identify_repo(question)

    if repo is None:
        return {
            "answer": (
                "Je n'ai pas pu identifier de quel dépôt GitHub tu parles. "
                "Dépôts connus : " + ", ".join(KNOWN_REPOS.values())
            ),
            "sources": [],
        }

    try:
        info = fetch_repo_info(repo)
    except httpx.HTTPStatusError as e:
        return {
            "answer": f"Impossible de récupérer les informations pour '{repo}' (erreur GitHub : {e.response.status_code}).",
            "sources": [],
        }
    except httpx.RequestError as e:
        return {
            "answer": f"Erreur réseau en contactant l'API GitHub : {e}",
            "sources": [],
        }

    prompt = (
        f"Voici les données à jour récupérées depuis l'API GitHub pour le dépôt {info['full_name']} :\n\n"
        f"- Description : {info['description']}\n"
        f"- Étoiles : {info['stars']}\n"
        f"- Forks : {info['forks']}\n"
        f"- Issues ouvertes : {info['open_issues']}\n"
        f"- Dernière release : {info['latest_release_tag']} (publiée le {info['latest_release_date']})\n"
        f"- Notes de la dernière release : {info['latest_release_notes']}\n\n"
        f"Question de l'utilisateur : {question}\n\n"
        "Réponds en français, de façon concise, en te basant uniquement sur ces données."
    )

    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [info["url"]],
    }


def print_result(question: str, result: dict):
    print(f"\nQuestion : {question}\n")
    print("Réponse :")
    print(result["answer"])
    if result["sources"]:
        print("\nSources utilisées :")
        for src in result["sources"]:
            print(f"  - {src}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage : python agents/github_agent.py "ta question ici"')
        sys.exit(1)

    q = " ".join(sys.argv[1:])
    result = answer_question(q)
    print_result(q, result)