"""
Supervisor : décide quel(s) agent(s) déclencher via structured output.
Testable en standalone (nécessite MISTRAL_API_KEY dans .env).
"""
import os
from typing import List, Literal

from pydantic import BaseModel, Field
from langchain_mistralai import ChatMistralAI

from app.src.state import AgentState

AGENT_NAMES = Literal[
    "rag_docs",
    "web_research",
    "github",
    "contacts",
    "applications",
]

AGENT_DESCRIPTIONS = """
- rag_docs: documentation officielle Mistral (API, modèles, quickstart, pricing, fenêtre de contexte...)
- web_research: papers académiques (Arxiv) et actualités/articles récents sur Mistral AI
- github: recherche de dépôts GitHub liés à Mistral AI (org mistralai ou communauté)
- contacts: informations de contact (support, sales, communauté, réseaux sociaux)
- applications: cas d'usage, cookbooks, exemples concrets d'utilisation de Mistral AI
""".strip()


class RoutingDecision(BaseModel):
    """Décision de routage : liste des agents à déclencher pour répondre à la question."""

    agents: List[AGENT_NAMES] = Field(
        description=(
            "Liste des agents pertinents à appeler pour répondre à la question. "
            "Une question composite peut nécessiter plusieurs agents en parallèle. "
            "Ne renvoie que les agents réellement utiles."
        )
    )


SUPERVISOR_SYSTEM_PROMPT = f"""Tu es le superviseur d'un système multi-agent expert de Mistral AI.
Voici les agents disponibles et leur spécialité :

{AGENT_DESCRIPTIONS}

Analyse la question de l'utilisateur et sélectionne UNIQUEMENT les agents pertinents.
Si la question est composite (ex: "quels sont les derniers modèles ET comment les contacter"),
sélectionne plusieurs agents pour qu'ils travaillent en parallèle.
Ne sélectionne jamais un agent qui n'apporterait rien à la réponse."""


def get_supervisor_llm():
    model = ChatMistralAI(
        model="mistral-large-latest",
        api_key=os.environ.get("MISTRAL_API_KEY"),
        temperature=0,
    )
    return model.with_structured_output(RoutingDecision)


def supervisor_node(state: AgentState) -> dict:
    question = state["messages"][-1].content
    llm = get_supervisor_llm()
    decision: RoutingDecision = llm.invoke(
        [
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("human", question),
        ]
    )
    return {"next_agent": decision.agents}


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    fake_state = {
        "messages": [
            HumanMessage(
                "Quels sont les derniers modèles Mistral et comment contacter le support ?"
            )
        ]
    }
    print(supervisor_node(fake_state))