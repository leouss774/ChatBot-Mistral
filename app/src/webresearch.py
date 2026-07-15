"""
STUB — à remplacer par le vrai code de Hazem.
Contrat : def web_research_node(state) -> {"web_results": [str]}
"""
from app.src.state import AgentState


def web_research_node(state: AgentState) -> dict:
    question = state["messages"][-1].content
    return {"web_results": [f"[STUB web_research] réponse simulée pour: {question}"]}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    fake_state = {"messages": [HumanMessage("derniers papers sur Mistral 7B")]}
    print(web_research_node(fake_state))