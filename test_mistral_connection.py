"""
Script de vérification rapide : confirme que la clé API Mistral fonctionne
avant de commencer à construire les agents.

Usage:
    1. Copie .env.example en .env et renseigne MISTRAL_API_KEY
    2. pip install -r requirements.txt
    3. python test_mistral_connection.py
"""

from mistralai.client import Mistral
from config import MISTRAL_API_KEY, MISTRAL_MODEL_CHAT

def main():
    if not MISTRAL_API_KEY:
        print("Erreur: MISTRAL_API_KEY manquante. Vérifie ton fichier .env")
        return

    client = Mistral(api_key=MISTRAL_API_KEY)

    response = client.chat.complete(
        model=MISTRAL_MODEL_CHAT,
        messages=[
            {"role": "user", "content": "Réponds juste par: connexion réussie"}
        ]
    )

    print("Réponse du modèle:", response.choices[0].message.content)

if __name__ == "__main__":
    main()