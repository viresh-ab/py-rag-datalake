import streamlit as st
import subprocess
from rag import ask
from vector_store import load_index, list_all_sources_with_counts

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Data Lake RAG",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Data Lake RAG")
st.caption("Chat with your OneDrive CASE_STUDIES documents")

# ---------------------------------
# Vector DB status
# ---------------------------------
with st.expander("📦 Vector DB Status", expanded=False):
    try:
        index, meta = load_index()
        st.success(f"Vector DB ready • {index.ntotal} chunks indexed")
    except Exception:
        st.error("Vector DB not found. Run ingestion.")

# ---------------------------------
# Ingestion section
# ---------------------------------
with st.expander("🔄 Run OneDrive Ingestion", expanded=False):
    if st.button("Run Ingestion"):
        with st.spinner("Ingesting documents from OneDrive..."):
            result = subprocess.run(
                ["python", "ingest.py"],
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
# Helpers (INTENT DETECTION)
# ---------------------------------
def is_listing_intent(text):
    t = text.lower().strip()
    phrases = [
        "list all case studies",
        "list all case study files",
        "list all files",
        "show all case studies",
        "what case studies are available",
        "files in my drive"
    ]
    return any(p in t for p in phrases)


def is_existence_intent(text):
    t = text.lower()
    phrases = [
        "is there any",
        "are there any",
        "do we have",
        "any case study related to"
    ]
    return any(p in t for p in phrases)


def detect_industry(text):
    t = text.lower()
    if "fashion" in t:
        return "fashion"
    if "fmcg" in t:
        return "fmcg"
    return None


def detect_explicit_document(text, all_sources):
    t = text.lower()
    for src in all_sources:
        if src.lower().replace(".pdf", "") in t:
            return src
    return None

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
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    sources_map = list_all_sources_with_counts()
    all_sources = list(sources_map.keys())

    # -----------------------------
    # 1️⃣ LISTING INTENT (NO LLM)
    # -----------------------------
    if is_listing_intent(prompt):
        response = "### 📁 Case Studies in Your Data Lake\n\n"
        for s in sorted(all_sources):
            response += f"- {s}\n"

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": all_sources
        })
        st.stop()

    # -----------------------------
    # 2️⃣ EXISTENCE INTENT (NO LLM)
    # -----------------------------
    if is_existence_intent(prompt):
        industry = detect_industry(prompt)
        matched = []

        if industry:
            matched = [s for s in all_sources if industry in s.lower()]

        if matched:
            response = (
                f"Yes. There {'is' if len(matched)==1 else 'are'} "
                f"{len(matched)} {industry.capitalize()} case "
                f"{'study' if len(matched)==1 else 'studies'} available:\n\n"
            )
            for m in matched:
                response += f"- {m}\n"
        else:
            response = f"No case studies related to the {industry.capitalize()} industry are available."

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": matched
        })
        st.stop()

    # -----------------------------
    # 3️⃣ DOCUMENT-SPECIFIC INTENT
    # -----------------------------
    explicit_doc = detect_explicit_document(prompt, all_sources)

    if explicit_doc:
        answer, _ = ask(
            prompt,
            required_keyword=explicit_doc.replace(".pdf", "")
        )

        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("📄 Sources"):
                st.write(f"- {explicit_doc}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": [explicit_doc]
        })
        st.stop()

    # -----------------------------
    # 4️⃣ NORMAL RAG (FILTERED)
    # -----------------------------
    industry = detect_industry(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, srcs = ask(prompt, required_keyword=industry)
            st.markdown(answer)

            if srcs:
                with st.expander("📄 Sources"):
                    for s in srcs:
                        st.write(f"- {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": srcs
    })
