"""
Contrat commun de l'équipe (Camelia, Hazem, Oussema, Wiem).
NE PAS MODIFIER sans prévenir toute l'équipe — tous les agents en dépendent.
"""
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: List[str]
    rag_results: List[str]
    web_results: List[str]
    github_results: List[str]
    contacts_results: List[str]
    applications_results: List[str]
    final_answer: str
