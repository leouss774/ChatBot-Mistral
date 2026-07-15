"""
rag_agent.py — Agent RAG simple

Principe :
1. Reçoit une question de l'utilisateur
2. Récupère les chunks les plus pertinents dans Chroma (retrieval)
3. Construit un prompt avec ces chunks comme contexte
4. Demande au modèle de chat Mistral de répondre UNIQUEMENT à partir de ce
   contexte, en citant ses sources
5. Affiche la réponse + la liste des sources utilisées

Usage :
    python agents/rag_agent.py "Comment installer le client Python Mistral ?"

Ou en mode interactif (sans argument) :
    python agents/rag_agent.py
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
CHAT_MODEL = "mistral-small-latest"
TOP_K = 5

SYSTEM_PROMPT = """Tu es un assistant qui répond à des questions sur Mistral AI \
(le modèle, l'API, le SDK Python) en te basant UNIQUEMENT sur les extraits de \
documentation fournis ci-dessous.

Règles strictes :
- Réponds uniquement à partir des extraits fournis. Si les extraits ne \
contiennent pas l'information demandée, dis-le clairement plutôt que d'inventer.
- Cite tes sources : après chaque affirmation, indique entre crochets le \
numéro de l'extrait correspondant, par exemple [1] ou [2, 3].
- Réponds en français, de façon claire et concise.
"""

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def retrieve(question: str, top_k: int = TOP_K):
    db = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    collection = db.get_collection(COLLECTION_NAME)

    query_embedding = client.embeddings.create(
        model=EMBED_MODEL, inputs=[question]
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(docs, metas)
    ]


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{i}] (source : {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(blocks)


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "Aucune information pertinente trouvée dans la base documentaire.",
            "sources": [],
        }

    context = build_context(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Extraits de documentation :\n\n{context}\n\nQuestion : {question}",
        },
    ]

    response = client.chat.complete(model=CHAT_MODEL, messages=messages)
    answer = response.choices[0].message.content

    sources = sorted({chunk["source"] for chunk in chunks})

    return {"answer": answer, "sources": sources}


def print_result(question: str, result: dict):
    print(f"\nQuestion : {question}\n")
    print("Réponse :")
    print(result["answer"])
    print("\nSources utilisées :")
    for src in result["sources"]:
        print(f"  - {src}")
    print()


def run_cli():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = answer_question(question)
        print_result(question, result)
    else:
        print("Agent RAG — tape 'exit' pour quitter.\n")
        while True:
            question = input("Ta question : ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            result = answer_question(question)
            print_result(question, result)


if __name__ == "__main__":
    run_cli()