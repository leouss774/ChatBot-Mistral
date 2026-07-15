"""
Construction du graphe complet.
Rien à changer ici quand Hazem/Oussema livrent leurs vrais agents : il suffit
de remplacer le contenu des fichiers stubs (app/src/agents/*.py) par leur
vrai code, tant qu'ils respectent le contrat de app/src/state.py.
rag_docs est déjà branché sur le vrai agent de Camelia.
"""
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

from app.src.state import AgentState
from app.src.supervisor import supervisor_node
from app.src.agents.rag_docs import rag_docs_node
from app.src.agents.web_research import web_research_node
from app.src.agents.github_agent import github_node
from app.src.agents.contacts import contacts_node
from app.src.agents.applications import applications_node
from app.src.agents.synthesizer import synthesizer_node

# Mapping nom d'agent (utilisé par le supervisor) -> nom de nœud dans le graphe
AGENT_NODE_NAMES = {
    "rag_docs": "rag_docs",
    "web_research": "web_research",
    "github": "github",
    "contacts": "contacts",
    "applications": "applications",
}


def route_to_agents(state: AgentState):
    """
    Fonction de routage conditionnelle appelée après le supervisor.
    Utilise Send pour déclencher chaque agent choisi EN PARALLÈLE (fan-out).
    """
    selected = state.get("next_agent") or []
    if not selected:
        return "synthesizer"
    return [Send(AGENT_NODE_NAMES[name], state) for name in selected if name in AGENT_NODE_NAMES]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_docs", rag_docs_node)
    graph.add_node("web_research", web_research_node)
    graph.add_node("github", github_node)
    graph.add_node("contacts", contacts_node)
    graph.add_node("applications", applications_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_to_agents,
        ["rag_docs", "web_research", "github", "contacts", "applications", "synthesizer"],
    )

    for agent_node in ["rag_docs", "web_research", "github", "contacts", "applications"]:
        graph.add_edge(agent_node, "synthesizer")

    graph.add_edge("synthesizer", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


compiled_graph = build_graph()


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    app_graph = build_graph()
    config = {"configurable": {"thread_id": "test-thread-1"}}
    result = app_graph.invoke(
        {
            "messages": [HumanMessage("Quels sont les modèles Mistral disponibles ?")],
            "next_agent": [],
            "rag_results": [],
            "web_results": [],
            "github_results": [],
            "contacts_results": [],
            "applications_results": [],
            "final_answer": "",
        },
        config=config,
    )
    print(result["final_answer"])