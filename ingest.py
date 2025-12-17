from io import BytesIO
from openai import OpenAI
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from onedrive_client import (
    get_folder_id_by_name,
    list_pdfs_from_folder_id,
    download_pdf
)
from vector_store import add_vectors, reset_index

client = OpenAI()


def chunk_text(text, size=800, overlap=100):
    chunks = []
    step = size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + size])
    return chunks


def embed(texts):
    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [e.embedding for e in res.data]


def ingest():
    reset_index()
    # 1. Resolve OneDrive folder
    folder_id = get_folder_id_by_name("CASE_STUDIES")
    pdfs = list_pdfs_from_folder_id(folder_id)

    if not pdfs:
        raise ValueError("No PDFs found in OneDrive folder")

    all_chunks = []

    # 2. Read PDFs
    for pdf in pdfs:
        pdf_bytes = download_pdf(pdf["id"])
        reader = PdfReader(BytesIO(pdf_bytes))

        # ✅ FIXED LINE (properly closed)
        text = " ".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        if not text.strip():
            continue

        # 3. Chunk text
        for chunk in chunk_text(text):
            all_chunks.append({
                "text": chunk,
                "source": pdf["name"]
            })

    if not all_chunks:
        raise ValueError("PDFs found but no extractable text")

    # 4. Embed + store
    vectors = embed([c["text"] for c in all_chunks])
    add_vectors(vectors, all_chunks)

    # Windows-safe print
    print(f"Ingested {len(all_chunks)} chunks from OneDrive")


if __name__ == "__main__":
    ingest()

