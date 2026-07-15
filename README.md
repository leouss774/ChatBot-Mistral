# Mistral Knowledge Bot

Backend de départ pour un chatbot RAG sur Mistral AI, avec récupération live par requête, indexation Chroma et génération via l'API Mistral.

## Ce que fait cette base

- détecte un agent selon la question: docs, web research, GitHub, contacts, applications
- récupère du contenu live via HTTP à chaque requête
- découpe le texte en chunks et les indexe dans Chroma
- vectorise les chunks avec `mistral-embed`
- répond avec `mistral-large-latest` en citant les sources

## Mise en route

1. Créer un environnement Python.
2. Installer les dépendances:

```bash
pip install -r requirements.txt
```

3. Définir les variables d'environnement:

```bash
set MISTRAL_API_KEY=your_key
set GITHUB_TOKEN=optional_token
```

4. Lancer l'API:

```bash
uvicorn app.main:app --reload
```

## Endpoints

- `GET /health`
- `POST /chat`
- `POST /ingest`

### Exemple `POST /chat`

```json
{
  "question": "Quels sont les modèles Mistral et leurs usages ?",
  "top_k": 5
}
```

## Étapes suivantes recommandées

- ajouter un crawler plus fin pour `docs.mistral.ai` avec sitemap
- brancher Tavily/Serper pour compléter les recherches web
- enrichir l'agent Contacts avec pages équipe/presse officielles
- ajouter une UI simple en Streamlit ou Next.js
