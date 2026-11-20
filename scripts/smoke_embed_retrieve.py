"""Minimal embed-store-retrieve loop (SCOPE Suggested Approach steps 2-3)."""

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data" / "chroma_smoke"


def main() -> None:
    """Smoke-test embed → store → query against a temporary Chroma path."""
    DB_DIR.mkdir(parents=True, exist_ok=True)

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        client.delete_collection("smoke")
    except Exception:  # noqa: BLE001, S110
        pass
    collection = client.create_collection(name="smoke", embedding_function=embed_fn)

    texts = [
        "Remote employees receive a home-office stipend of $75 per month.",
        "Travel meals are capped at $50 per person per day without manager approval.",
    ]
    collection.add(ids=["t1", "t2"], documents=texts)

    query = "What is the daily meal reimbursement limit for travel?"
    result = collection.query(query_texts=[query], n_results=2)

    print("Query:", query)
    print("Top match id:", result["ids"][0][0])
    print("Top match text:", result["documents"][0][0])
    print("Distances:", result["distances"][0])

    assert result["ids"][0][0] == "t2", "Expected meal policy text to rank first"
    print("OK: retrieval returned the more relevant text")


if __name__ == "__main__":
    main()