from openai import OpenAI
from vector_store import search

client = OpenAI()

# =========================
# CONFIG
# =========================
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"

TOP_K = 8                   # fetch more chunks, filter strictly later
SIMILARITY_THRESHOLD = 0.78 # strong relevance only
MIN_CHUNKS_PER_SOURCE = 2   # 🔥 KEY RULE
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

    # 🔍 Retrieve chunks
    results = search(q_vec, top_k=TOP_K)

    texts = []
    source_stats = {}

    for r in results:
        score = r.get("score", 0)
        if score < SIMILARITY_THRESHOLD:
            continue

        text = r.get("text", "")
        source = r.get("source")

        if text:
            texts.append(text)

        if not source:
            continue

        # Track contribution per document
        if source not in source_stats:
            source_stats[source] = {
                "count": 0,
                "max_score": 0
            }

        source_stats[source]["count"] += 1
        source_stats[source]["max_score"] = max(
            source_stats[source]["max_score"], score
        )

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
                    "Write clearly and professionally. "
                    "Use ONLY the provided context. "
                    "If context is insufficient, say so."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    # =========================
    # 🔥 STRICT SOURCE SELECTION
    # =========================
    # Only keep documents that contributed
    # at least 2 strong chunks
    filtered_sources = {
        src: stats
        for src, stats in source_stats.items()
        if stats["count"] >= MIN_CHUNKS_PER_SOURCE
    }

    # Rank by strength
    sources = sorted(
        filtered_sources,
        key=lambda s: filtered_sources[s]["max_score"],
        reverse=True
    )[:MAX_SOURCES]

    return response.choices[0].message.content, sources
