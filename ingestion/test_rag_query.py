"""
test_rag_query.py — pose une question, récupère les chunks les plus proches
depuis Chroma, et affiche leurs sources pour vérifier la pertinence du RAG.

Usage :
    python ingestion/test_rag_query.py "Comment installer le client Python Mistral ?"
"""

import os
import sys
import chromadb
from dotenv import load_dotenv
from mistralai.client import Mistral
load_dotenv()

VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "chatbot_mistral"
EMBED_MODEL = "mistral-embed"
TOP_K = 3

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def run(question: str):
    db = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection = db.get_collection(COLLECTION_NAME)

    query_embedding = client.embeddings.create(
        model=EMBED_MODEL, inputs=[question]
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )

    print(f"Question : {question}\n")
    print(f"Top {TOP_K} extraits pertinents :\n")

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        print(f"--- Extrait {i + 1} (source: {meta['source']}, distance: {dist:.4f}) ---")
        print(doc[:400] + ("..." if len(doc) > 400 else ""))
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage : python ingestion/test_rag_query.py "ta question ici"')
        sys.exit(1)

    run(" ".join(sys.argv[1:]))