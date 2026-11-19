"""Show hybrid (vector + keyword) vs vector-only on an exact policy code."""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.config import get_settings

QUERY = "EX-MEAL-50"


def main() -> None:
    settings = get_settings()
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    col = chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(
        settings.collection_name, embedding_function=embed_fn
    )

    all_rows = col.get(include=["documents", "metadatas"])
    kw = [
        (_id, doc, meta)
        for _id, doc, meta in zip(all_rows["ids"], all_rows["documents"], all_rows["metadatas"])
        if QUERY.lower() in doc.lower()
    ]

    vec = col.query(query_texts=[QUERY], n_results=settings.top_k)
    vec_ids, vec_docs = vec["ids"][0], vec["documents"][0]

    print("Query:", QUERY)
    print("\nVector-only:")
    for i, (_id, doc) in enumerate(zip(vec_ids, vec_docs), 1):
        print(f"  {i}. {_id}: {doc[:90].replace(chr(10), ' ')}")

    print("\nKeyword hits:")
    for _id, doc, meta in kw:
        print(f"  - {_id} ({meta.get('document_name')})")

    hybrid = []
    for _id, _, _ in kw:
        if _id not in hybrid:
            hybrid.append(_id)
    for _id in vec_ids:
        if _id not in hybrid:
            hybrid.append(_id)

    print("\nHybrid (keyword first, then vector):")
    for i, _id in enumerate(hybrid[: settings.top_k], 1):
        print(f"  {i}. {_id}")

    in_vec = any(QUERY in d for d in vec_docs)
    in_kw = bool(kw)
    print("\nExact code in vector top-k?", in_vec)
    print("Exact code found by keyword?", in_kw)
    if in_kw:
        print("OK: hybrid can surface EX-MEAL-50 via keyword")


if __name__ == "__main__":
    main()