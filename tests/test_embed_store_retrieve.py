import pytest
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


@pytest.mark.slow
def test_minimal_embed_store_retrieve(tmp_path: Path):
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.create_collection(name="smoke", embedding_function=embed_fn)

    collection.add(
        ids=["t1", "t2"],
        documents=[
            "Remote employees receive a home-office stipend of $75 per month.",
            "Travel meals are capped at $50 per person per day without manager approval.",
        ],
    )

    result = collection.query(
        query_texts=["What is the daily meal reimbursement limit for travel?"],
        n_results=2,
    )
    assert result["ids"][0][0] == "t2"