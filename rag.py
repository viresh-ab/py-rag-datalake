from openai import OpenAI
from vector_store import search

client = OpenAI()

def embed_query(query):
    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    )
    return res.data[0].embedding

def ask(question, top_k=5):
    q_vec = embed_query(question)
    results = search(q_vec, top_k)

    texts = []
    sources = set()

    for r in results:
        # ✅ NEW format (dict)
        if isinstance(r, dict):
            texts.append(r.get("text", ""))
            if "source" in r:
                sources.add(r["source"])
        # ✅ OLD format (string)
        elif isinstance(r, str):
            texts.append(r)

    context = "\n\n".join(texts)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Answer only using the context"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    return response.choices[0].message.content, sorted(sources)
