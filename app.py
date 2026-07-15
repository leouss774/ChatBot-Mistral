"""
app.py — Interface Streamlit pour le système Q&A multi-agents Mistral AI

Usage :
    streamlit run app.py
"""

import streamlit as st
from agents.router_agent import answer_question

# --- Configuration de la page ---
st.set_page_config(
    page_title="Mistral Q&A",
    page_icon="🌬️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Métadonnées d'affichage par agent ---
AGENT_INFO = {
    "rag": {"label": "Documentation & Recherche", "icon": "📚", "color": "#4F8BF9"},
    "github": {"label": "GitHub", "icon": "🐙", "color": "#6E5494"},
    "web_search": {"label": "Recherche Web", "icon": "🌐", "color": "#22C55E"},
    "contact": {"label": "Contact & Support", "icon": "✉️", "color": "#F59E0B"},
}

EXAMPLE_QUESTIONS = [
    "Comment installer le client Python Mistral ?",
    "Quels sont les derniers commits sur mistral-inference ?",
    "Quelles sont les nouveautés Mistral cette semaine ?",
    "Comment contacter le support Mistral ?",
]

# --- CSS personnalisé ---
st.markdown("""
<style>
    .agent-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-bottom: 10px;
    }
    .source-link {
        display: block;
        padding: 6px 10px;
        margin: 4px 0;
        background-color: rgba(128, 128, 128, 0.08);
        border-radius: 6px;
        font-size: 0.85rem;
        text-decoration: none;
    }
    .source-link:hover {
        background-color: rgba(128, 128, 128, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# --- État de session ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Sidebar ---
with st.sidebar:
    st.header("🌬️ Mistral Q&A")
    st.caption("Assistant multi-agents sur l'écosystème Mistral AI")

    st.markdown("---")
    st.subheader("Agents disponibles")
    for key, info in AGENT_INFO.items():
        st.markdown(
            f"<span class='agent-badge' style='background-color:{info['color']}'>"
            f"{info['icon']} {info['label']}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Exemples de questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# --- Titre principal ---
st.title("🌬️ Mistral Q&A")
st.caption("Pose une question sur Mistral AI — documentation, code, GitHub, actualités ou contact.")

# --- Affichage de l'historique ---
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        info = AGENT_INFO.get(entry["category"], {"label": entry["category"], "icon": "🤖", "color": "#888"})
        st.markdown(
            f"<span class='agent-badge' style='background-color:{info['color']}'>"
            f"{info['icon']} {info['label']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(entry["answer"])
        if entry["sources"]:
            with st.expander(f"📎 {len(entry['sources'])} source(s)"):
                for src in entry["sources"]:
                    st.markdown(f"<a class='source-link' href='{src}' target='_blank'>🔗 {src}</a>", unsafe_allow_html=True)

# --- Question en attente (clic sur exemple) ---
pending = st.session_state.pop("pending_question", None)

# --- Champ de saisie ---
question = st.chat_input("Écris ta question ici...") or pending

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            try:
                result = answer_question(question)
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.stop()

        info = AGENT_INFO.get(result["category"], {"label": result["category"], "icon": "🤖", "color": "#888"})
        st.markdown(
            f"<span class='agent-badge' style='background-color:{info['color']}'>"
            f"{info['icon']} {info['label']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(f"📎 {len(result['sources'])} source(s)"):
                for src in result["sources"]:
                    st.markdown(f"<a class='source-link' href='{src}' target='_blank'>🔗 {src}</a>", unsafe_allow_html=True)

    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "category": result["category"],
    })