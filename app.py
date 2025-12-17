import streamlit as st
import subprocess
import sys
from rag import ask
from vector_store import load_index

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
page_title="Markelytics Solutions | Datalake",
page_icon="🌐",
layout="centered"
)
import streamlit as st
st.title("💬 Markelytics - Data Lake")
st.caption("Chat with your OneDrive CASE_STUDIES documents")

st.markdown(
    """
    <style>
    /* --- Hide Streamlit chrome --- */
    header { display: none !important; }
    footer { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }

    /* --- Fix root app layout --- */
    html, body, .stApp {
        height: 100%;
        margin: 0;
        padding: 0;
    }

    /* --- Fix main container --- */
    .stMainBlockContainer {
        max-width: 100% !important;
        padding: 1rem 2rem 6rem 2rem !important; /* bottom space for chat */
        min-height: 100vh;
        box-sizing: border-box;
    }

    /* --- Fix chat input sticking to bottom --- */
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: white;
        padding: 0.75rem 1rem;
        border-top: 1px solid #e5e7eb;
        z-index: 999;
    }

    /* --- Prevent content hiding behind chat --- */
    .stMainBlockContainer > div:last-child {
        margin-bottom: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


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
with st.expander("🔄 Run OneDrive Ingestion", expanded=False):
if st.button("Run Ingestion"):
with st.spinner("Ingesting documents from OneDrive..."):
result = subprocess.run(
[sys.executable, "ingest.py"],
capture_output=True,
text=True
)

if result.returncode == 0:
st.success("Ingestion completed successfully")
st.code(result.stdout)
else:
st.error("Ingestion failed")
st.code(result.stderr)

# ---------------------------------
# Chat session state
# ---------------------------------
if "messages" not in st.session_state:
st.session_state.messages = []

# ---------------------------------
# Render chat history
# ---------------------------------
for msg in st.session_state.messages:
with st.chat_message(msg["role"]):
st.markdown(msg["content"])

if msg["role"] == "assistant" and msg.get("sources"):
with st.expander("📄 Sources"):
for s in msg["sources"]:
st.write(f"- {s}")

# ---------------------------------
# Chat input
# ---------------------------------
prompt = st.chat_input("Ask a question about your case studies...")

if prompt:
# Show user message
st.session_state.messages.append({
"role": "user",
"content": prompt
})

with st.chat_message("user"):
st.markdown(prompt)

# Generate assistant response
with st.chat_message("assistant"):
with st.spinner("Thinking..."):
try:
answer, sources = ask(prompt)
st.markdown(answer)

if sources:
with st.expander("📄 Sources"):
for s in sources:
st.write(f"- {s}")

# Save assistant message
st.session_state.messages.append({
"role": "assistant",
"content": answer,
"sources": sources
})

except Exception as e:
error_msg = f"❌ Error: {str(e)}"
st.error(error_msg)
st.session_state.messages.append({
"role": "assistant",
"content": error_msg
})

