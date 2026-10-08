"""
Simple web UI for the Self-Corrective RAG pipeline, built with Streamlit.

Run with:
    streamlit run streamlit_app.py

This gives you a browser-based interface instead of the command line:
- A text box to type a question
- The final answer, clearly displayed
- An expandable trace showing every step the pipeline took (retrieve,
  grade, generate, hallucination check, answer check, retries)
- Status indicators for whether the answer passed grounding/relevance checks
"""
import streamlit as st

from src.config import settings
from src.graph import run
from src.vectorstore import build_vectorstore
from src.ingestion import load_and_chunk

st.set_page_config(page_title="Self-Corrective RAG", page_icon="🔎", layout="centered")

st.title("🔎 Self-Corrective RAG")
st.caption(
    "Hallucination-resistant question answering: every answer is graded for "
    "evidence relevance, factual grounding, and whether it actually answers "
    "your question -- before you see it."
)

# --- Sidebar: ingestion controls ---
with st.sidebar:
    st.header("Documents")
    source_dir = st.text_input("Source folder", value="data/sample_docs")
    if st.button("Re-ingest documents", use_container_width=True):
        with st.spinner(f"Loading and embedding documents from {source_dir} ..."):
            try:
                settings.validate()
                chunks = load_and_chunk(source_dir)
                build_vectorstore(chunks)
                st.success(f"Ingested {len(chunks)} chunk(s) from {source_dir}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.caption(f"LLM provider: **{settings.llm_provider}**")
    st.caption(f"Generator model: **{settings.generator_model}**")
    st.caption(f"Grader model: **{settings.grader_model}**")
    st.caption(f"Retrieval k: **{settings.retrieval_k}**")
    st.caption(f"Max retries: **{settings.max_retries}**")

# --- Main: ask a question ---
question = st.text_input("Ask a question about your documents", placeholder="e.g. What triggers the query rewrite step?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    try:
        settings.validate()
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    with st.spinner("Running retrieval, grading, generation, and verification..."):
        try:
            result = run(question)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong: {exc}")
            st.stop()

    st.subheader("Answer")
    st.markdown(result.get("generation", "(no answer generated)"))

    # Status badges
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Retries used", result.get("retries", 0))
    with col2:
        grounded = result.get("hallucination_ok")
        st.metric("Grounded in evidence", "✅ Yes" if grounded else ("❌ No" if grounded is False else "—"))
    with col3:
        answers = result.get("answers_question")
        st.metric("Answers the question", "✅ Yes" if answers else ("❌ No" if answers is False else "—"))

    if result.get("used_web_search"):
        st.info("This answer was supplemented with web search results.")

    with st.expander("Execution trace (step-by-step)"):
        for i, line in enumerate(result.get("trace", []), start=1):
            st.text(f"{i}. {line}")

    with st.expander("Retrieved / graded documents"):
        docs = result.get("documents", [])
        if not docs:
            st.write("No documents were graded relevant for the final answer.")
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            st.markdown(f"**{i}. {source}**")
            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

elif ask_clicked:
    st.warning("Please type a question first.")
