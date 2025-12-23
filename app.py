import streamlit as st
from rag import ask

st.set_page_config(
    page_title="Markelytics Data Lake",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Markelytics Data Lake")
st.caption("Fast, accurate responses powered by embeddings and LLMs")

# =========================
# CHAT INPUT
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


# ---------------------------------
# Vector DB status
# ---------------------------------
# with st.expander("📦 Vector DB Status", expanded=False):
#     try:
#         index, meta = load_index()
#         st.success(f"Vector DB ready • {index.ntotal} chunks indexed")
#     except Exception:
#         st.error("Vector DB not found. Run ingestion.")

# ---------------------------------
# Ingestion section
# ---------------------------------
# with st.expander("🔄 Run OneDrive Ingestion", expanded=False):
#     if st.button("Run Ingestion"):
#         with st.spinner("Ingesting documents from OneDrive..."):
#             result = subprocess.run(
#                    [sys.executable, "ingest.py"],
#                 capture_output=True,
#                 text=True
#             )

#             if result.returncode == 0:
#                 st.success("Ingestion completed successfully")
#                 st.code(result.stdout)
#             else:
#                 st.error("Ingestion failed")
#                 st.code(result.stderr)

# ---------------------------------
# Chat session state
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------
# Render chat history
# ---------------------------------
