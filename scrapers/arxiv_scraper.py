"""
Récupère les métadonnées et l'abstract des papiers listés dans
config.ARXIV_PAPER_IDS via l'API arXiv (gratuite, sans clé).
Sauvegarde dans data/raw/papers/.

Note: ce script récupère les métadonnées + abstract, pas le PDF complet.
L'extraction du texte complet du PDF sera ajoutée à une étape ultérieure
si besoin (benchmarks détaillés dans le corps du papier).

Usage:
    python scrapers/arxiv_scraper.py
"""

import os
import sys
import json
import time
import requests
import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ARXIV_PAPER_IDS, DATA_RAW_DIR

OUTPUT_DIR = os.path.join(DATA_RAW_DIR, "papers")
ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_paper(arxiv_id: str) -> dict | None:
    resp = requests.get(ARXIV_API, params={"id_list": arxiv_id}, timeout=15)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entry = root.find("atom:entry", NS)
    if entry is None:
        return None

    title = entry.find("atom:title", NS).text.strip()
    summary = entry.find("atom:summary", NS).text.strip()
    published = entry.find("atom:published", NS).text
    authors = [a.find("atom:name", NS).text for a in entry.findall("atom:author", NS)]
    pdf_link = next(
        (l.get("href") for l in entry.findall("atom:link", NS) if l.get("title") == "pdf"),
        None,
    )

    return {
        "arxiv_id": arxiv_id,
        "source_type": "paper",
        "title": title,
        "authors": authors,
        "published": published,
        "abstract": summary,
        "pdf_url": pdf_link,
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not ARXIV_PAPER_IDS:
        print("config.ARXIV_PAPER_IDS est vide. Ajoute des IDs arXiv (ex: '2310.06825') avant de lancer ce script.")
        return

    print(f"Récupération de {len(ARXIV_PAPER_IDS)} papier(s)...")

    for arxiv_id in ARXIV_PAPER_IDS:
        print(f"- {arxiv_id}")
        try:
            data = fetch_paper(arxiv_id)
        except requests.RequestException as e:
            print(f"  Erreur sur {arxiv_id}: {e}")
            continue

        if data is None:
            print(f"  Papier introuvable: {arxiv_id}")
            continue

        filepath = os.path.join(OUTPUT_DIR, f"{arxiv_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  -> sauvegardé: {filepath}")

        time.sleep(1)  # rate-limit poli envers l'API arXiv

    print("Terminé.")


if __name__ == "__main__":
    main()