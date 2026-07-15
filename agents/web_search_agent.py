"""
web_search_agent.py — Agent de recherche web
"""

import os
import sys
from ddgs import DDGS
from dotenv import load_dotenv
from mistralai.client import Mistral
from agents.lang_utils import language_instruction

load_dotenv()

CHAT_MODEL = "mistral-small-latest"
MAX_RESULTS = 5

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def build_search_query(question: str) -> str:
    if "mistral" in question.lower():
        return f"Mistral AI {question}"
    return f"Mistral AI (entreprise IA) {question}"


def search_web(query: str, max_results: int = MAX_RESULTS) -> list[dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in results
    ]


def build_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] {r['title']}\n{r['snippet']}\n(source : {r['url']})")
    return "\n\n".join(blocks)


def answer_question(question: str) -> dict:
    query = build_search_query(question)

    try:
        results = search_web(query)
    except Exception as e:
        return {"answer": f"Erreur lors de la recherche web : {e}", "sources": []}

    if not results:
        return {"answer": "Aucun résultat de recherche web trouvé pour cette question.", "sources": []}

    context = build_context(results)

    prompt = (
        f"Voici des résultats de recherche web récents :\n\n{context}\n\n"
        f"Question : {question}\n\n"
        "Réponds de façon concise, en te basant sur ces résultats. "
        "Cite le numéro de la source entre crochets après chaque affirmation, ex. [1]. "
        "Si les résultats ne permettent pas de répondre avec certitude, dis-le."
        + language_instruction(question)
    )

    response = client.chat.complete(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    sources = [r["url"] for r in results]

    return {"answer": response.choices[0].message.content, "sources": sources}


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
        print('Usage : python agents/web_search_agent.py "ta question ici"')
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    result = answer_question(q)
    print_result(q, result)