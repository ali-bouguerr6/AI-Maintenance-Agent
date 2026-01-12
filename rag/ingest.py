import os
import glob
from typing import List
import chromadb
from pypdf import PdfReader
import requests

BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

COLLECTION_NAME = "maintenance_docs"


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 1100, overlap: int = 200) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += step
    return chunks


def ollama_embed(text: str) -> List[float]:
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120
    )
    r.raise_for_status()
    return r.json()["embedding"]


def ingest():
    pdf_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    if not pdf_files:
        raise RuntimeError(f"Aucun PDF trouvé dans {DOCS_DIR}")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    ids, docs, metas, embeds = [], [], [], []

    for pdf_path in pdf_files:
        source = os.path.basename(pdf_path)
        print(f"📄 Lecture: {source}")

        text = extract_pdf_text(pdf_path)
        chunks = chunk_text(text)

        if not chunks:
            print(f"⚠️  Aucun texte extrait: {source}")
            continue

        for idx, ch in enumerate(chunks):
            doc_id = f"{source}::chunk{idx}"
            ids.append(doc_id)
            docs.append(ch)
            metas.append({"source": source, "chunk": idx})
            embeds.append(ollama_embed(ch))

    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    print("\n✅ Ingestion terminée")
    print(f"PDFs indexés: {len(pdf_files)}")
    print(f"Chunks indexés: {len(ids)}")
    print(f"Store: {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
