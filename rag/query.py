import os
import chromadb
import requests
from typing import List, Dict, Tuple

BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_store")

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

COLLECTION_NAME = "maintenance_docs"


def ollama_embed(text: str) -> List[float]:
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120
    )
    r.raise_for_status()
    return r.json()["embedding"]


def rag_search(query: str, k: int = 3) -> List[Dict]:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    q_emb = ollama_embed(query)
    res = col.query(query_embeddings=[q_emb], n_results=k)

    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "source": res["metadatas"][0][i].get("source", "unknown"),
            "text": res["documents"][0][i]
        })
    return hits


def format_rag_context(hits: List[Dict]) -> Tuple[str, List[str]]:
    """
    Transforme les résultats RAG en:
    - rag_context: texte à injecter dans le prompt LLM
    - sources: liste des noms de fichiers sources (sans doublons)
    """
    if not hits:
        return "", []

    sources: List[str] = []
    blocks: List[str] = []

    for h in hits:
        src = h.get("source", "unknown")
        if src not in sources:
            sources.append(src)

        txt = (h.get("text") or "").strip()
        if txt:
            blocks.append(f"[SOURCE: {src}]\n{txt}")

    rag_context = "\n\n".join(blocks).strip()
    return rag_context, sources


if __name__ == "__main__":
    q = "Que faire en cas de toolwear_anomaly ?"
    hits = rag_search(q, k=3)

    rag_context, sources = format_rag_context(hits)

    print("\n📚 Sources:", ", ".join(sources) if sources else "Aucune")
    print("\n--- RAG CONTEXT (aperçu) ---")
    print(rag_context[:1200], "...")
