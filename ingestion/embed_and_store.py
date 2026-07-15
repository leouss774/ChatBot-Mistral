"""
embed_and_store.py

Lit data/processed/chunks.jsonl (produit par chunker.py), génère un embedding
par chunk via l'API Mistral (mistral-embed), et stocke le tout dans une base
Chroma persistante (vectorstore/).
"""

import os
import json
import chromadb
from dotenv import load_dotenv
from mistralai.client import Mistral
load_dotenv()

CHUNKS_PATH = "data/processed/chunks.jsonl"
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "chatbot_mistral"

EMBED_MODEL = "mistral-embed"
BATCH_SIZE = 16

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, inputs=batch)
        vectors.extend([d.embedding for d in resp.data])
        print(f"  embeddings {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    return vectors


def run():
    chunks = load_chunks()
    print(f"{len(chunks)} chunks chargés depuis {CHUNKS_PATH}")

    if not chunks:
        print("Aucun chunk à traiter — lance d'abord chunker.py")
        return

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"source": c["source"], "n_words": c["n_words"]} for c in chunks]

    print("Génération des embeddings...")
    embeddings = embed_texts(texts)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    db = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    # repart d'une collection propre à chaque ingestion complète
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = db.create_collection(COLLECTION_NAME)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\n{len(chunks)} chunks stockés dans {VECTORSTORE_DIR}/ (collection '{COLLECTION_NAME}')")


if __name__ == "__main__":
    run()