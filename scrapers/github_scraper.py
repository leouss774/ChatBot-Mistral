"""
Récupère les infos clés des repos GitHub listés dans config.GITHUB_REPOS :
README, dernières releases, description, stars.
Sauvegarde dans data/raw/github/.

Usage:
    python scrapers/github_scraper.py
"""

import os
import sys
import json
import time
import base64
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GITHUB_REPOS, GITHUB_TOKEN, DATA_RAW_DIR

OUTPUT_DIR = os.path.join(DATA_RAW_DIR, "github")
API_BASE = "https://api.github.com"

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"


def get_json(url: str):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def get_readme_text(repo: str) -> str:
    data = get_json(f"{API_BASE}/repos/{repo}/readme")
    if not data:
        return ""
    content = data.get("content", "")
    try:
        return base64.b64decode(content).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def scrape_repo(repo: str) -> dict:
    print(f"- {repo}")
    repo_info = get_json(f"{API_BASE}/repos/{repo}") or {}
    releases = get_json(f"{API_BASE}/repos/{repo}/releases?per_page=5") or []
    readme = get_readme_text(repo)

    return {
        "repo": repo,
        "source_type": "github",
        "description": repo_info.get("description"),
        "stars": repo_info.get("stargazers_count"),
        "url": repo_info.get("html_url"),
        "readme_text": readme,
        "latest_releases": [
            {
                "tag": r.get("tag_name"),
                "name": r.get("name"),
                "published_at": r.get("published_at"),
                "body": r.get("body"),
            }
            for r in releases
        ],
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not GITHUB_TOKEN:
        print("Attention: pas de GITHUB_TOKEN dans .env -> limite de 60 requêtes/heure.")

    print(f"Récupération de {len(GITHUB_REPOS)} repo(s)...")

    for repo in GITHUB_REPOS:
        try:
            data = scrape_repo(repo)
        except requests.RequestException as e:
            print(f"  Erreur sur {repo}: {e}")
            continue

        filename = repo.replace("/", "__") + ".json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  -> sauvegardé: {filepath}")

        time.sleep(1)

    print("Terminé.")


if __name__ == "__main__":
    main()