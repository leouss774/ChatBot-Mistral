"""
STUB — à remplacer par le vrai code de Oussema.
Contrat : def contacts_node(state) -> {"contacts_results": [str]}
"""
from app.src.state import AgentState


def contacts_node(state: AgentState) -> dict:
    question = state["messages"][-1].content
    return {"contacts_results": [f"[STUB contacts] réponse simulée pour: {question}"]}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    fake_state = {"messages": [HumanMessage("comment contacter le support Mistral ?")]}
    print(contacts_node(fake_state))