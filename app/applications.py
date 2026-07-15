"""
STUB — à remplacer par le vrai code de Oussema.
Contrat : def applications_node(state) -> {"applications_results": [str]}
"""
from app.src.state import AgentState


def applications_node(state: AgentState) -> dict:
    question = state["messages"][-1].content
    return {"applications_results": [f"[STUB applications] réponse simulée pour: {question}"]}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    fake_state = {"messages": [HumanMessage("cas d'usage de function calling")]}
    print(applications_node(fake_state))