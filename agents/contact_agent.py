"""
contact_agent.py — Agent Contact

Répond aux questions sur comment contacter/joindre Mistral AI (support,
ventes entreprise, presse, sécurité...). Ces informations changent rarement,
donc on utilise une base de connaissances statique vérifiée plutôt qu'une
recherche web à chaque appel — plus rapide et plus fiable.

Source des informations : pages officielles mistral.ai/contact et
help.mistral.ai, vérifiées manuellement.

Usage :
    python agents/contact_agent.py "Comment contacter le support Mistral AI pour un usage entreprise ?"
"""

import os
import sys
from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()

CHAT_MODEL = "mistral-small-latest"

# Base de connaissances statique — à mettre à jour périodiquement si les
# canaux de contact de Mistral changent.
CONTACT_KNOWLEDGE = """
Canaux de contact officiels de Mistral AI :

- Support produit (Le Chat, Studio, API, etc.) : via le widget de support intégré \
sur help.mistral.ai. Les utilisateurs Enterprise passent par le même widget, mais \
leurs demandes sont automatiquement priorisées et routées vers un workflow dédié.

- Ventes entreprise / usage professionnel : formulaire "Contact sales" disponible \
sur mistral.ai (page Services et pages produits). C'est le point d'entrée pour les \
déploiements enterprise, la personnalisation de modèles, et les partenariats.

- Presse et événements : press@mistral.ai

- Demandes liées à la vie privée / données personnelles : via la plateforme Mistral \
(formulaire dédié aux droits sur les données).

- Signalement de vulnérabilité de sécurité : programme de divulgation responsable \
("vulnerability disclosure program"), accessible depuis mistral.ai/contact. Les bugs \
généraux (non sécurité) passent par les canaux de support standards.

- Communauté / entraide : serveur Discord officiel de Mistral AI.

- Contact légal général : contact@mistral.ai (adresse indiquée sur les mentions légales).

- Page de contact centrale pour tout sujet ne rentrant pas dans les catégories \
ci-dessus : mistral.ai/contact
"""

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def answer_question(question: str) -> dict:
    prompt = (
        f"Base de connaissances sur les canaux de contact Mistral AI :\n\n"
        f"{CONTACT_KNOWLEDGE}\n\n"
        f"Question : {question}\n\n"
        "Réponds en français, de façon concise et directe, en indiquant le ou les "
        "canaux de contact pertinents pour cette question précise. Si la question "
        "porte sur un sujet non couvert par cette base, dis-le et oriente vers la "
        "page de contact générale mistral.ai/contact."
    )

    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": ["https://mistral.ai/contact/", "https://help.mistral.ai"],
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
        print('Usage : python agents/contact_agent.py "ta question ici"')
        sys.exit(1)

    q = " ".join(sys.argv[1:])
    result = answer_question(q)
    print_result(q, result)