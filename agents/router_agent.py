"""
router_agent.py — Routeur multi-agents

Point d'entrée unique du système Q&A. Classe automatiquement chaque question
dans une catégorie (rag, github, web_search, contact), puis délègue à l'agent
spécialisé correspondant.

Catégories :
    - rag        : documentation, papiers de recherche, exemples de code
                    → agents/rag_agent.py (base vectorielle Chroma)
    - github     : infos à jour sur un dépôt GitHub (release, stars...)
                    → agents/github_agent.py (API GitHub en direct)
    - web_search : actualités, annonces récentes, infos non couvertes ailleurs
                    → agents/web_search_agent.py (recherche DuckDuckGo)
    - contact    : comment joindre Mistral AI (support, ventes, presse...)
                    → agents/contact_agent.py (base de connaissances statique)

Usage :
    python agents/router_agent.py "Comment installer le client Python Mistral ?"

Ou en mode interactif :
    python agents/router_agent.py
"""

import os
import sys
import json
import re
from dotenv import load_dotenv
from mistralai.client import Mistral

from agents import rag_agent, github_agent, web_search_agent, contact_agent

load_dotenv()

ROUTER_MODEL = "mistral-small-latest"

ROUTER_SYSTEM_PROMPT = """Tu es un classifieur de questions pour un système Q&A sur l'écosystème Mistral AI.

Classe la question de l'utilisateur dans EXACTEMENT une de ces 4 catégories :

- "rag" : questions sur la documentation Mistral, les papiers de recherche \
(architecture des modèles, benchmarks, scores), ou demandant des exemples de \
code / usage du SDK Python. Ce sont des informations stables, déjà documentées.

- "github" : questions sur un dépôt GitHub précis de Mistral (dernière release, \
nombre d'étoiles, statut d'un projet open-source).

- "web_search" : questions sur des annonces récentes, actualités, nouveaux \
modèles, ou toute info qui nécessite une recherche web en temps réel.

- "contact" : questions sur comment contacter/joindre Mistral AI (support, \
ventes entreprise, presse, sécurité, vie privée).

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant/après, au format :
{"category": "<rag|github|web_search|contact>"}
"""

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

AGENTS = {
    "rag": rag_agent,
    "github": github_agent,
    "web_search": web_search_agent,
    "contact": contact_agent,
}


def classify(question: str) -> str:
    response = client.chat.complete(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
        category = parsed.get("category", "rag")
    except json.JSONDecodeError:
        category = "rag"  # repli sûr : le RAG est le cas d'usage le plus courant

    if category not in AGENTS:
        category = "rag"

    return category


def answer_question(question: str) -> dict:
    category = classify(question)
    agent = AGENTS[category]
    result = agent.answer_question(question)
    result["category"] = category
    return result


def print_result(question: str, result: dict):
    print(f"\nQuestion : {question}")
    print(f"[Agent utilisé : {result['category']}]\n")
    print("Réponse :")
    print(result["answer"])
    if result["sources"]:
        print("\nSources utilisées :")
        for src in result["sources"]:
            print(f"  - {src}")
    print()


def run_cli():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = answer_question(question)
        print_result(question, result)
    else:
        print("Système Q&A Mistral — tape 'exit' pour quitter.\n")
        while True:
            question = input("Ta question : ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            result = answer_question(question)
            print_result(question, result)


if __name__ == "__main__":
    run_cli()