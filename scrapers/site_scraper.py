"""
Scraper pour le site officiel et la documentation Mistral.
Parcourt les URLs listées dans config.SITE_URLS et sauvegarde le texte
propre de chaque page dans data/raw/site/.

Usage:
    python scrapers/site_scraper.py
"""

import os
import re
import sys
import json
import time
import hashlib
import requests
from bs4 import BeautifulSoup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SITE_URLS, DATA_RAW_DIR

OUTPUT_DIR = os.path.join(DATA_RAW_DIR, "site")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MistralQABot/1.0)"}


def clean_text(html: str) -> str:
    """Extrait le texte lisible d'une page HTML, sans scripts/styles/nav."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def url_to_filename(url: str) -> str:
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", url).strip("-")[:60]
    return f"{slug}-{h}.json"


def scrape_page(url: str) -> dict | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Erreur sur {url}: {e}")
        return None

    return {
        "url": url,
        "source_type": "site",
        "text": clean_text(resp.text),
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Scraping de {len(SITE_URLS)} URL(s)...")

    for url in SITE_URLS:
        print(f"- {url}")
        data = scrape_page(url)
        if data is None:
            continue

        filepath = os.path.join(OUTPUT_DIR, url_to_filename(url))
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  -> sauvegardé: {filepath} ({len(data['text'])} caractères)")

        time.sleep(1)  # politesse envers le serveur

    print("Terminé.")


if __name__ == "__main__":
    main()