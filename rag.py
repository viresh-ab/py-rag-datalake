from openai import OpenAI
from vector_store import search
import re

client = OpenAI()

# =========================
# CONFIG
# =========================
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4.1-mini"

TOP_K = 12
SIMILARITY_THRESHOLD = 0.78
MIN_CHUNKS_PER_SOURCE = 2
MIN_TOPIC_MATCHES = 2      # 🔥 NEW
MAX_SOURCES = 2            # keep blogs clean


# =========================
# TOPIC KEYWORDS
# =========================
TOPIC_KEYWORDS = [
    "shopper insight",
    "shopper insights",
    "consumer behavior",
    "purchase decision",
    "buying behavior",
    "path to purchase"
]


def topic_match_score(text: str) -> int:
    """Count how many topic keywords appear in text"""
    text = text.lower()
    return sum(1 for kw in TOPIC_KEYWORDS if kw in text)


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

    results = search(q_vec, top_k=TOP_K)

    texts = []
    source_stats = {}

    for r in results:
        score = r.get("score", 0)
        if score < SIMILARITY_THRESHOLD:
            continue

        text = r.get("text", "")
        source = r.get("source")

        if not text or not source:
            continue

        topic_hits = topic_match_score(text)

        if topic_hits == 0:
            continue  # ❌ chunk is not about shopper insights

        texts.append(text)

        if source not in source_stats:
            source_stats[source] = {
                "chunk_count": 0,
                "topic_hits": 0,
                "max_score": 0
            }

        source_stats[source]["chunk_count"] += 1
        source_stats[source]["topic_hits"] += topic_hits
        source_stats[source]["max_score"] = max(
            source_stats[source]["max_score"], score
        )

    # =========================
    # STRICT SOURCE FILTERING
    # =========================
    filtered_sources = {
        src: stats
        for src, stats in source_stats.items()
        if stats["chunk_count"] >= MIN_CHUNKS_PER_SOURCE
        and stats["topic_hits"] >= MIN_TOPIC_MATCHES
    }

    # Rank by topic dominance, then similarity
    sources = sorted(
        filtered_sources,
        key=lambda s: (
            filtered_sources[s]["topic_hits"],
            filtered_sources[s]["max_score"]
        ),
        reverse=True
    )[:MAX_SOURCES]

    context = "\n\n".join(texts)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a market research expert. "
                    "Write a professional blog. "
                    "Use ONLY the provided context. "
                    "If the context is insufficient, say so clearly."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )

    return response.choices[0].message.content, sources

