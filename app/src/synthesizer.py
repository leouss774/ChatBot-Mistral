"""
STUB — à remplacer par le vrai code de Oussema.
Contrat : def synthesizer_node(state) -> {"final_answer": str}
Fusionne tous les résultats non-vides en une réponse finale.
"""
from app.src.state import AgentState


def synthesizer_node(state: AgentState) -> dict:
    parts = []
    for key in (
        "rag_results",
        "web_results",
        "github_results",
        "contacts_results",
        "applications_results",
    ):
        values = state.get(key) or []
        if values:
            parts.extend(values)

    if not parts:
        parts = ["[STUB synthesizer] aucun résultat disponible."]

    final = "[STUB synthesizer] Réponse fusionnée :\n" + "\n".join(f"- {p}" for p in parts)
    return {"final_answer": final}


if __name__ == "__main__":
    fake_state = {
        "rag_results": ["Mistral Large est le modèle phare."],
        "web_results": [],
        "github_results": ["mistralai/mistral-inference (5k stars)"],
        "contacts_results": [],
        "applications_results": [],
    }
    print(synthesizer_node(fake_state))