import streamlit as st
import subprocess
import os
from rag import ask

st.set_page_config(
    page_title="Markelytics Data Lake",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Markelytics Data Lake")
st.caption("Fast, accurate responses powered by embeddings and LLMs")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "data", "faiss", "index.faiss")

# =========================
# INGESTION BUTTON
# =========================
if not os.path.exists(FAISS_PATH):
    st.warning("Vector index not found. Please run ingestion first.")

    if st.button("🚀 Run Ingestion"):
        with st.spinner("Running ingestion... this may take a few minutes"):
            result = subprocess.run(
                ["python", "ingest.py"],
                capture_output=True,
                text=True
            )

        if result.returncode == 0:
            st.success("Ingestion completed successfully. Reloading app...")
            st.experimental_rerun()
        else:
            st.error("Ingestion failed")
            st.code(result.stderr)

    st.stop()

# =========================
# CHAT UI
# =========================
question = st.chat_input("Ask a question about your case studies...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            answer, sources = ask(question)
            st.write(answer)

            if sources:
                st.markdown("### 📄 Sources")
                for s in sources:
                    st.markdown(f"- **{s}**")

        except RuntimeError as e:
            st.error(str(e))
