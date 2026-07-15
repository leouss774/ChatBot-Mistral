"""
STUB — à remplacer par le vrai code de Hazem.
Contrat : def github_node(state) -> {"github_results": [str]}
"""
from app.src.state import AgentState


def github_node(state: AgentState) -> dict:
    question = state["messages"][-1].content
    return {"github_results": [f"[STUB github] réponse simulée pour: {question}"]}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    fake_state = {"messages": [HumanMessage("repos officiels mistralai")]}
    print(github_node(fake_state))