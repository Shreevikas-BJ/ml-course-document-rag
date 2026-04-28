import os
import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.retrieval.retrieve import Retriever
from src.rag.rag_chain import StrictRAG, REFUSAL_MESSAGE
from src.utils.langsmith_setup import enable_langsmith


# -------------------------
# Load Streamlit secrets → env
# -------------------------
def load_secrets_into_env():
    """
    Works in BOTH:
    - Streamlit Cloud (st.secrets exists)
    - Local runs (no secrets.toml) without crashing

    Priority:
    1) existing OS env vars (setx)
    2) secrets.toml (if present)
    """
    # If already set via OS env, do nothing
    if os.getenv("OPENAI_API_KEY") and os.getenv("LANGCHAIN_API_KEY"):
        return

    # Only access st.secrets if a secrets.toml exists in known locations
    candidate_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.cwd() / "app" / ".streamlit" / "secrets.toml",
    ]
    has_secrets_file = any(p.exists() for p in candidate_paths)

    if not has_secrets_file:
        return  # local run without secrets.toml → no crash

    # Now it's safe to touch st.secrets
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    if "LANGCHAIN_API_KEY" in st.secrets:
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

    if "LANGCHAIN_ENDPOINT" in st.secrets:
        os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGCHAIN_ENDPOINT"]

    if "LANGCHAIN_PROJECT" in st.secrets:
        os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

    if "LANGCHAIN_TRACING_V2" in st.secrets:
        os.environ["LANGCHAIN_TRACING_V2"] = str(st.secrets["LANGCHAIN_TRACING_V2"]).lower()

load_secrets_into_env()


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="ML RAG Doc Q&A", layout="wide")

st.title("📄 ML CourseWork RAG Document Q&A")
st.caption(
    "CPU runtime • FAISS retrieval • Strict grounding • Refuses when documents don't support the answer"
)

# Sidebar
st.sidebar.header("Settings")

sim_threshold = st.sidebar.slider(
    "Similarity threshold (gate)",
    0.0,
    0.8,
    0.60,
    0.01,
)

top_k = st.sidebar.selectbox("Top-K retrieval", [3, 5, 8, 10], index=1)

openai_model = st.sidebar.selectbox(
    "OpenAI model",
    ["gpt-4o-mini", "gpt-5-mini"],
    index=0,
)

st.sidebar.markdown("---")
enable_tracing = st.sidebar.toggle("Enable LangSmith tracing", value=True)

if enable_tracing:
    enable_langsmith(project_name="ml-rag-docqa-phase1")
    st.sidebar.success("Tracing enabled")

# -------------------------
# Load system (cached)
# -------------------------
@st.cache_resource(show_spinner=True)
def load_system(sim_threshold: float, top_k: int, openai_model: str):
    retriever = Retriever(
        sim_threshold=sim_threshold,
        top_k=top_k,
        device="cpu",
    )
    rag = StrictRAG(
        retriever=retriever,
        openai_model=openai_model,
        temperature=0.0,
    )
    return retriever, rag


# -------------------------
# Main interaction
# -------------------------
question = st.text_input(
    "Ask a question about the ML PDFs:",
    placeholder="e.g., What is explainable machine learning?",
)

col1, col2 = st.columns([1, 1])
with col1:
    ask = st.button("Answer", type="primary")
with col2:
    clear = st.button("Clear")

if clear:
    st.session_state.clear()
    st.rerun()

if ask:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set.")
        st.stop()

    retriever, rag = load_system(sim_threshold, top_k, openai_model)

    retrieved, best_score, passed = retriever.search(question)

    st.subheader("Retrieval")
    st.write(f"**Best cosine similarity:** `{best_score:.4f}`")
    st.write(f"**Passed threshold:** `{passed}`")

    if not passed or not retrieved:
        st.warning(REFUSAL_MESSAGE)
        with st.expander("Retrieved chunks (debug)"):
            for c in retrieved:
                st.markdown(
                    f"- **{c.score:.4f}** — {c.pdf_name} p.{c.page_number} (`{c.chunk_id}`)"
                )
        st.stop()

    out = rag.answer(question)

    st.subheader("Answer")
    st.markdown(out.answer)

    st.subheader("Sources")
    for c in out.retrieved:
        with st.expander(
            f"{c.score:.4f} — {c.pdf_name} p.{c.page_number} — {c.chunk_id}"
        ):
            st.write(c.text)
