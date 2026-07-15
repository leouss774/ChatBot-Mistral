"""
Agent RAG Docs — répond aux questions sur la documentation officielle Mistral,
le site officiel, et les modèles existants.

Contrat: def rag_docs_node(state: AgentState) -> dict
Retourne uniquement {"rag_results": [str]}

Ce module est testable en standalone (voir bloc __main__ en bas), sans
dépendre du graphe LangGraph complet ni des autres agents de l'équipe.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
PERSIST_DIR = str(ROOT_DIR / "data" / "chroma_db")
COLLECTION_NAME = "mistral_docs"

TOP_K = 5

SYSTEM_PROMPT = """Tu es un expert de la documentation officielle de Mistral AI \
(docs.mistral.ai, mistral.ai, liste des modèles).

Règles strictes :
- Réponds UNIQUEMENT à partir du contexte fourni ci-dessous, issu de la documentation officielle.
- Si le contexte ne contient pas l'information demandée, dis-le clairement plutôt que d'inventer.
- Cite systématiquement tes sources en fin de réponse sous la forme : Sources: [nom_fichier1, nom_fichier2].
- Sois précis et concis, donne les noms exacts de modèles/produits quand c'est pertinent.

Contexte (extraits de la documentation officielle) :
{context}
"""

_vectorstore = None
_llm = None


def _get_vectorstore() -> Chroma:
    """Charge le vector store persisté (lazy loading, une seule fois par process)."""
    global _vectorstore
    if _vectorstore is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=api_key)
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    return _vectorstore


def _get_llm() -> ChatMistralAI:
    global _llm
    if _llm is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        _llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=api_key)
    return _llm


def _format_context(docs) -> str:
    parts = []
    for d in docs:
        source = d.metadata.get("source", "inconnu")
        parts.append(f"[source: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def rag_docs_node(state: dict) -> dict:
    """
    Node LangGraph pour l'agent RAG Docs.

    Args:
        state: AgentState avec au minimum state["messages"][-1].content = question

    Returns:
        dict avec la clé "rag_results": [réponse_avec_sources: str]
    """
    question = state["messages"][-1].content

    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    if not docs:
        return {"rag_results": ["Aucune information pertinente trouvée dans la documentation indexée."]}

    context = _format_context(docs)
    llm = _get_llm()

    response = llm.invoke([
        ("system", SYSTEM_PROMPT.format(context=context)),
        ("user", question),
    ])

    return {"rag_results": [response.content]}


if __name__ == "__main__":
    # Test standalone — ne dépend pas du graphe LangGraph ni des autres agents.
    from langchain_core.messages import HumanMessage

    test_questions = [
        "Quels sont les modèles Mistral disponibles actuellement ?",
        "Quelle est la différence entre Mistral Large 3 et Mistral Small 4 ?",
        "Comment faire un appel API avec function calling ?",
        "Quel modèle utiliser pour de l'OCR ?",
        "Quelle est la licence de Voxtral TTS ?",
        "Où trouver le Discord officiel de Mistral ?",
    ]

    for q in test_questions:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        fake_state = {"messages": [HumanMessage(content=q)]}
        result = rag_docs_node(fake_state)
        print(result["rag_results"][0])
