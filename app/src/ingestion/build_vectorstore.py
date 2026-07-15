"""
Script d'ingestion du corpus Mistral -> vector store Chroma.

Usage:
    python -m src.ingestion.build_vectorstore

Prérequis:
    - MISTRAL_API_KEY dans l'environnement (ou fichier .env à la racine)
    - Corpus markdown présent dans data/raw/*.md

Ce script :
1. Charge tous les fichiers .md de data/raw/
2. Les découpe en chunks (chunking markdown-aware : on essaie de couper
   sur les titres ## / ### avant de couper sur les paragraphes)
3. Génère les embeddings avec le modèle mistral-embed
4. Persiste le tout dans un Chroma local (data/chroma_db)
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PERSIST_DIR = str(ROOT_DIR / "data" / "chroma_db")
COLLECTION_NAME = "mistral_docs"


def load_corpus() -> list:
    """Charge tous les .md du dossier data/raw en tant que Documents LangChain."""
    if not RAW_DATA_DIR.exists() or not any(RAW_DATA_DIR.glob("*.md")):
        print(f"[ERREUR] Aucun fichier .md trouvé dans {RAW_DATA_DIR}")
        print("Ajoute au moins un fichier markdown dans data/raw/ avant de lancer l'ingestion.")
        sys.exit(1)

    loader = DirectoryLoader(
        str(RAW_DATA_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    # Ajoute une métadonnée "source" propre (nom de fichier) pour la citation
    for doc in docs:
        filename = Path(doc.metadata.get("source", "unknown")).name
        doc.metadata["source"] = filename
        doc.metadata["type"] = "doc"

    print(f"[OK] {len(docs)} document(s) chargé(s) depuis {RAW_DATA_DIR}")
    return docs


def split_documents(docs: list) -> list:
    """Découpe les documents en chunks, en essayant de respecter la structure markdown."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"[OK] {len(chunks)} chunk(s) généré(s) (chunk_size=800, overlap=120)")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    """Génère les embeddings mistral-embed et persiste dans Chroma."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("[ERREUR] Variable d'environnement MISTRAL_API_KEY manquante.")
        print("Ajoute-la dans un fichier .env à la racine du projet : MISTRAL_API_KEY=xxx")
        sys.exit(1)

    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)

    print(f"[...] Génération des embeddings et écriture dans {PERSIST_DIR} (peut prendre quelques dizaines de secondes)")
    start = time.time()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )

    elapsed = time.time() - start
    print(f"[OK] Vector store construit et persisté en {elapsed:.1f}s -> {PERSIST_DIR}")
    return vectorstore


def main():
    print("=== Ingestion du corpus Mistral AI ===")
    docs = load_corpus()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("=== Terminé. Le vector store est prêt pour rag_docs.py ===")


if __name__ == "__main__":
    main()
