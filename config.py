"""
Configuration centrale du projet mistral-qa-agent.
Toutes les clés/API sont lues depuis les variables d'environnement (.env).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Mistral ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL_CHAT = "mistral-large-latest"     # modèle de génération pour les agents
MISTRAL_MODEL_EMBED = "mistral-embed"           # modèle d'embeddings pour le RAG

# --- GitHub ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")    # optionnel mais recommandé (évite le rate-limit anonyme)

# --- Sources à couvrir (à compléter à l'étape 1) ---
SITE_URLS = [
    "https://mistral.ai",
    "https://docs.mistral.ai",
]

GITHUB_REPOS = [
    "mistralai/mistral-inference",
    "mistralai/client-python",
    "mistralai/cookbook",
]

ARXIV_PAPER_IDS = [
    "2310.06825",  # Mistral 7B
    "2401.04088",  # Mixtral of Experts
]

# --- Chemins locaux ---
DATA_RAW_DIR = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
VECTORSTORE_DIR = "vectorstore"
EVAL_QUESTIONS_PATH = "eval/test_questions.json"