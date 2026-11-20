"""Rerank vector candidates with a CrossEncoder."""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder

from rag.config import get_settings

QUERY = "What is the current travel meal limit?"


def main() -> None:
    settings = get_settings()
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    collection = chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(
        settings.collection_name, embedding_function=embed_fn
    )

    result = collection.query(query_texts=[QUERY], n_results=settings.candidate_n)
    ids, docs = result["ids"][0], result["documents"][0]

    print("Query:", QUERY)
    print("\nBefore rerank:")
    for index, (chunk_id, doc) in enumerate(zip(ids, docs), 1):
        print(f"  {index}. {chunk_id}: {doc[:80].replace(chr(10), ' ')}")

    reranker = CrossEncoder(settings.cross_encoder_model)
    scores = reranker.predict([[QUERY, doc] for doc in docs])
    ranked = sorted(
        zip(ids, docs, scores), key=lambda row: float(row[2]), reverse=True
    )

    print("\nAfter rerank:")
    for index, (chunk_id, doc, score) in enumerate(ranked[: settings.top_k], 1):
        print(
            f"  {index}. {float(score):.3f} {chunk_id}: "
            f"{doc[:80].replace(chr(10), ' ')}"
        )

    print("\nOK: reranked")


if __name__ == "__main__":
    main()