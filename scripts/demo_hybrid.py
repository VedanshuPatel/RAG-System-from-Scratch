"""Show hybrid (vector + keyword) vs vector-only on an exact policy code."""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.config import get_settings

QUERY = "EX-MEAL-50"


def main() -> None:
    settings = get_settings()
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    collection = chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(
        settings.collection_name, embedding_function=embed_fn
    )

    all_rows = collection.get(include=["documents", "metadatas"])
    keyword_hits = [
        (chunk_id, doc, meta)
        for chunk_id, doc, meta in zip(
            all_rows["ids"], all_rows["documents"], all_rows["metadatas"]
        )
        if QUERY.lower() in doc.lower()
    ]

    vec = collection.query(query_texts=[QUERY], n_results=settings.top_k)
    vec_ids, vec_docs = vec["ids"][0], vec["documents"][0]

    print("Query:", QUERY)
    print("\nVector-only:")
    for index, (chunk_id, doc) in enumerate(zip(vec_ids, vec_docs), 1):
        print(f"  {index}. {chunk_id}: {doc[:90].replace(chr(10), ' ')}")

    print("\nKeyword hits:")
    for chunk_id, doc, meta in keyword_hits:
        print(f"  - {chunk_id} ({meta.get('document_name')})")

    hybrid = []
    for chunk_id, _, _ in keyword_hits:
        if chunk_id not in hybrid:
            hybrid.append(chunk_id)
    for chunk_id in vec_ids:
        if chunk_id not in hybrid:
            hybrid.append(chunk_id)

    print("\nHybrid (keyword first, then vector):")
    for index, chunk_id in enumerate(hybrid[: settings.top_k], 1):
        print(f"  {index}. {chunk_id}")

    in_vec = any(QUERY in doc for doc in vec_docs)
    in_kw = bool(keyword_hits)
    print("\nExact code in vector top-k?", in_vec)
    print("Exact code found by keyword?", in_kw)
    if in_kw:
        print("OK: hybrid can surface EX-MEAL-50 via keyword")


if __name__ == "__main__":
    main()
