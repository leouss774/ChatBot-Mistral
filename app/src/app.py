"""
UI de démo Streamlit pour le graphe LangGraph.
Lancer depuis la racine du repo (là où se trouve le dossier app/) avec :
    streamlit run app/src/ui/app.py
"""
import sys
from pathlib import Path

# Permet de lancer `streamlit run app/src/ui/app.py` depuis la racine du repo
sys.path.append(str(Path(__file__).resolve().parents[3]))

import streamlit as st
from langchain_core.messages import HumanMessage

from app.src.graph import compiled_graph

AGENT_LABELS = {
    "supervisor": "🧭 Supervisor (routing)",
    "rag_docs": "📚 RAG Docs",
    "web_research": "🌐 Web Research",
    "github": "🐙 GitHub",
    "contacts": "📇 Contacts",
    "applications": "🛠️ Applications",
    "synthesizer": "✨ Synthesizer",
}

RESULT_KEYS = {
    "rag_docs": "rag_results",
    "web_research": "web_results",
    "github": "github_results",
    "contacts": "contacts_results",
    "applications": "applications_results",
}

st.set_page_config(page_title="Mistral AI Multi-Agent", page_icon="🤖", layout="wide")
st.title("🤖 Mistral AI Expert — Système Multi-Agent")
st.caption("LangGraph · Supervisor routing · Fan-out parallèle")

if "history" not in st.session_state:
    st.session_state.history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "demo-thread"

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Pose ta question sur Mistral AI...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state = {
        "messages": [HumanMessage(question)],
        "next_agent": [],
        "rag_results": [],
        "web_results": [],
        "github_results": [],
        "contacts_results": [],
        "applications_results": [],
        "final_answer": "",
    }

    agent_details = {}

    with st.chat_message("assistant"):
        status_box = st.status("Réflexion en cours...", expanded=True)
        final_answer = ""

        for update in compiled_graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in update.items():
                label = AGENT_LABELS.get(node_name, node_name)
                status_box.write(f"▶️ {label} a répondu")

                result_key = RESULT_KEYS.get(node_name)
                if result_key and node_output.get(result_key):
                    agent_details[node_name] = node_output[result_key]

                if node_name == "synthesizer":
                    final_answer = node_output.get("final_answer", "")

        status_box.update(label="Terminé", state="complete", expanded=False)

        st.markdown(final_answer or "_Pas de réponse générée._")

        if agent_details:
            with st.expander("🔍 Voir le détail par agent"):
                for node_name, values in agent_details.items():
                    st.markdown(f"**{AGENT_LABELS.get(node_name, node_name)}**")
                    for v in values:
                        st.markdown(f"- {v}")

    st.session_state.history.append({"role": "assistant", "content": final_answer})