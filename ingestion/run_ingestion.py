"""
run_ingestion.py — lance le pipeline complet d'ingestion :
  1. chunker.py       (découpage sémantique)
  2. embed_and_store.py (embeddings + stockage Chroma)
"""

from ingestion import chunker, embed_and_store


def run():
    print("=== Étape 1/2 : Chunking sémantique ===\n")
    chunker.run()

    print("\n=== Étape 2/2 : Embeddings + stockage vectoriel ===\n")
    embed_and_store.run()

    print("\n=== Ingestion terminée ===")


if __name__ == "__main__":
    run()