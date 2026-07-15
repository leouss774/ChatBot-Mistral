"""
evaluate.py — Évaluation automatique de l'agent RAG (LLM-as-judge)

Principe :
1. Charger les questions de test (eval/test_questions.json)
2. Faire répondre l'agent RAG à chaque question
3. Demander à Mistral de juger la réponse : est-elle correcte et utile ?
   La question est-elle dans le périmètre de la base documentaire actuelle
   (RAG) ou nécessiterait-elle un autre agent (GitHub live, web search, etc.) ?
4. Sauvegarder les résultats détaillés (eval/results.jsonl) et afficher un
   résumé par catégorie.

Usage :
    python -m eval.evaluate
"""

import os
import json
import re
from dotenv import load_dotenv
from mistralai.client import Mistral

from agents.rag_agent import answer_question

load_dotenv()

TEST_QUESTIONS_PATH = "eval/test_questions.json"
RESULTS_PATH = "eval/results.jsonl"

JUDGE_MODEL = "mistral-small-latest"

JUDGE_SYSTEM_PROMPT = """Tu es un évaluateur QA strict pour un agent RAG (retrieval-augmented generation) \
spécialisé sur l'écosystème Mistral AI (documentation, papiers de recherche, code du SDK Python).

On te donne une question, la réponse produite par l'agent, et les sources qu'il a utilisées.

Évalue :
1. "score" (entier 1 à 5) : qualité et exactitude de la réponse.
   - 5 = réponse complète, précise, bien sourcée
   - 3 = réponse partielle ou imprécise
   - 1 = réponse fausse, absente, ou l'agent a halluciné
2. "in_scope" (booléen) : la question porte-t-elle sur un sujet que la base \
documentaire actuelle (docs, papiers, README GitHub statiques) est censée \
couvrir ? Mets "false" si la question demande une info en temps réel \
(dernière release, dernière annonce) ou hors sujet (contact/support) que \
cette base ne peut structurellement pas avoir à jour.
3. "justification" (chaîne courte, 1-2 phrases, en français) : pourquoi ce score.

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant/après, au format :
{"score": <int>, "in_scope": <bool>, "justification": "<str>"}
"""

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def load_test_questions():
    with open(TEST_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def judge_answer(question: str, answer: str, sources: list[str]) -> dict:
    sources_str = ", ".join(sources) if sources else "(aucune)"
    user_content = (
        f"Question : {question}\n\n"
        f"Réponse de l'agent : {answer}\n\n"
        f"Sources utilisées : {sources_str}"
    )

    response = client.chat.complete(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Sécurité : parfois le modèle entoure le JSON de ```json ... ``` malgré la consigne
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"score": None, "in_scope": None, "justification": f"[Échec de parsing JSON] {raw[:200]}"}

    return parsed


def run():
    questions = load_test_questions()
    print(f"{len(questions)} question(s) de test chargée(s)\n")

    results = []

    with open(RESULTS_PATH, "w", encoding="utf-8") as out:
        for i, item in enumerate(questions, start=1):
            qid = item.get("id", f"q{i}")
            category = item.get("category", "inconnu")
            question = item["question"]

            print(f"[{i}/{len(questions)}] ({category}) {question}")

            rag_result = answer_question(question)
            judgment = judge_answer(question, rag_result["answer"], rag_result["sources"])

            record = {
                "id": qid,
                "category": category,
                "question": question,
                "answer": rag_result["answer"],
                "sources": rag_result["sources"],
                "score": judgment.get("score"),
                "in_scope": judgment.get("in_scope"),
                "justification": judgment.get("justification"),
            }
            results.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            score_display = record["score"] if record["score"] is not None else "?"
            scope_display = "dans le périmètre RAG" if record["in_scope"] else "hors périmètre RAG"
            print(f"    → score {score_display}/5 — {scope_display}")

    print(f"\nRésultats détaillés écrits dans {RESULTS_PATH}\n")
    print_summary(results)


def print_summary(results: list[dict]):
    print("=" * 60)
    print("RÉSUMÉ PAR CATÉGORIE")
    print("=" * 60)

    categories = sorted({r["category"] for r in results})

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        scored = [r["score"] for r in cat_results if isinstance(r["score"], (int, float))]
        avg_score = sum(scored) / len(scored) if scored else 0
        out_of_scope_count = sum(1 for r in cat_results if r["in_scope"] is False)

        print(f"\n{cat} ({len(cat_results)} question(s)) — score moyen : {avg_score:.1f}/5")
        if out_of_scope_count:
            print(f"  ⚠ {out_of_scope_count} question(s) jugée(s) hors périmètre du RAG actuel")
            print("    → besoin probable d'un agent dédié (GitHub live, web search, contact...)")

    all_scored = [r["score"] for r in results if isinstance(r["score"], (int, float))]
    global_avg = sum(all_scored) / len(all_scored) if all_scored else 0
    print(f"\n{'=' * 60}")
    print(f"SCORE GLOBAL : {global_avg:.1f}/5 sur {len(results)} question(s)")
    print("=" * 60)


if __name__ == "__main__":
    run()