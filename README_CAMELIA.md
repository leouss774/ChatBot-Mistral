# Camelia — RAG Docs Agent — Guide rapide (4h)

## Ce qui est déjà prêt dans ce dossier

- `data/raw/*.md` : corpus de départ (3 fichiers, contenu réel vérifié sur
  docs.mistral.ai et mistral.ai le jour du hackathon) :
  - `models_overview.md` — liste complète et à jour des modèles Mistral
  - `docs_structure_api.md` — structure de la doc, API, produits, pricing
  - `site_company_community.md` — entreprise, réseaux sociaux, contacts, Discord, GitHub
- `src/state.py` — le contrat `AgentState` partagé avec l'équipe (ne pas modifier)
- `src/ingestion/build_vectorstore.py` — script d'ingestion prêt à l'emploi
- `src/agents/rag_docs.py` — l'agent RAG, conforme au contrat `rag_docs_node(state) -> {"rag_results": [...]}`

## Étapes (dans l'ordre, ~2h30 max normalement)

### 1. Setup (10 min)

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
cp .env.example .env
# édite .env et mets ta clé API Mistral (https://console.mistral.ai)
```

### 2. Enrichir le corpus (30-45 min) — LE PLUS IMPORTANT

Le corpus de départ est volontairement minimal pour démarrer vite. Enrichis-le
en copiant/collant du contenu texte directement depuis :
- https://docs.mistral.ai/models/model-selection-guide (guide de choix de modèle)
- https://docs.mistral.ai/models/best-practices/prompt-engineering (bonnes pratiques)
- https://docs.mistral.ai/getting-started/quickstart (quickstart API)
- https://docs.mistral.ai/resources/cookbooks (liste des cookbooks)
- La page pricing officielle si tu veux des chiffres exacts

**Ne perds pas de temps à écrire un crawler générique.** Ouvre les pages,
copie le texte utile dans un nouveau fichier `.md` dans `data/raw/`, ajoute un
titre `# ...` et une ligne `Source: <url>` en haut de chaque fichier (comme
dans les fichiers existants) pour que la citation de sources reste correcte.

### 3. Lancer l'ingestion (2 min)

```bash
python -m src.ingestion.build_vectorstore
```

Ça doit afficher le nombre de documents chargés, le nombre de chunks générés,
et confirmer que le vector store est persisté dans `data/chroma_db/`.

Relance cette commande à chaque fois que tu ajoutes/modifies un fichier dans
`data/raw/`.

### 4. Tester l'agent RAG en standalone (5 min)

```bash
python -m src.agents.rag_docs
```

Ça va lancer 6 questions de test et afficher les réponses. Vérifie que :
- les réponses sont correctes et sourcées (`Sources: [...]` en fin de réponse)
- pas d'hallucination flagrante (si le contexte ne contient pas l'info, l'agent
  doit le dire plutôt qu'inventer — vérifie le prompt système dans `rag_docs.py`
  si ce n'est pas le cas)

### 5. Itérer sur la qualité (reste du temps)

- Si les réponses sont mauvaises → vérifie d'abord le chunking (des chunks trop
  gros/petits cassent souvent la pertinence) et le nombre de résultats récupérés
  (`TOP_K` dans `rag_docs.py`, actuellement 5)
- Si t'as le temps : ajoute un reranking simple (garder les 3 chunks les plus
  pertinents parmi les 8 récupérés, via un second appel LLM léger)
- Teste avec des questions "pièges" : questions hors sujet, questions sur des
  modèles très récents non présents dans le corpus, questions ambiguës

## Livrable pour l'intégration (Wiem)

Le seul fichier dont Wiem a besoin pour brancher ton agent dans le graphe :

```python
from src.agents.rag_docs import rag_docs_node
```

C'est une fonction pure `state -> {"rag_results": [str]}`, aucune dépendance
aux autres agents. Assure-toi juste que `data/chroma_db/` existe (donc que
l'ingestion a été lancée au moins une fois) avant l'intégration finale.

## Si tu bloques

- **Erreur clé API** → vérifie `.env` et que `python-dotenv` charge bien le fichier
  (le script `rag_docs.py` fait `load_dotenv()` en haut)
- **Chroma qui plante** → supprime `data/chroma_db/` et relance l'ingestion
  (`rm -rf data/chroma_db`)
- **Pas le temps d'enrichir le corpus** → pas grave, le corpus de départ (3
  fichiers) suffit pour une démo correcte sur les questions modèles/API/contact
