from openai import OpenAI
from vector_store import search

client = OpenAI()

# =========================
# CONFIG
# =========================
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"

TOP_K = 5                  # fetch only top relevant chunks
SIMILARITY_THRESHOLD = 0.75
MAX_SOURCES = 3


# =========================
# EMBEDDING
# =========================
def embed_query(query: str):
    res = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )
    return res.data[0].embedding


# =========================
# ASK (RAG PIPELINE)
# =========================
def ask(question: str):
    q_vec = embed_query(question)

    # 🔍 Retrieve relevant chunks
    results = search(q_vec, top_k=TOP_K)

    texts = []
    source_scores = {}

    for r in results:

        # ✅ NEW FORMAT (dict with score)
        if isinstance(r, dict):
            score = r.get("score", 0)

            # ❌ Ignore weak matches
            if score < SIMILARITY_THRESHOLD:
                continue

            text = r.get("text", "")
            source = r.get("source")

            if text:
                texts.append(text)

            # ✅ Track best score per source
            if source:
                if source not in source_scores or score > source_scores[source]:
                    source_scores[source] = score

        # ⚠️ OLD FORMAT (fallback – no scoring)
        elif isinstance(r, str):
            texts.append(r)

    # 🧠 Build context
    context = "\n\n".join(texts)

    # 🤖 LLM call
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a market research expert. "
                    "Answer ONLY using the provided context. "
                    "If the context is insufficient, say so clearly."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    # 📄 Sort & limit sources by relevance
    sources = sorted(
        source_scores,
        key=lambda x: source_scores[x],
        reverse=True
    )[:MAX_SOURCES]

    return response.choices[0].message.content, sources
