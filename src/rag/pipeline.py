"""Main RAG flow: hybrid retrieve → rerank → generate."""

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI
from sentence_transformers import CrossEncoder

from rag.config import Settings, get_settings


def get_collection(settings: Settings | None = None):
    settings = settings or get_settings()
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    return chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(
        name=settings.collection_name,
        embedding_function=embed_fn,
    )


def retrieve_hybrid(collection, question: str, candidate_n: int):
    vec = collection.query(query_texts=[question], n_results=candidate_n)
    vec_ids, vec_docs, vec_metas = vec["ids"][0], vec["documents"][0], vec["metadatas"][0]
    by_id = {i: (d, m) for i, d, m in zip(vec_ids, vec_docs, vec_metas)}

    all_rows = collection.get(include=["documents", "metadatas"])
    q = question.lower()
    ordered_ids = []
    for _id, doc, meta in zip(all_rows["ids"], all_rows["documents"], all_rows["metadatas"]):
        if q in doc.lower():
            by_id[_id] = (doc, meta)
            if _id not in ordered_ids:
                ordered_ids.append(_id)
    for _id in vec_ids:
        if _id not in ordered_ids:
            ordered_ids.append(_id)

    return ordered_ids, [by_id[i][0] for i in ordered_ids], [by_id[i][1] for i in ordered_ids]


def rerank(question, ids, docs, metas, model_name: str, top_k: int):
    reranker = CrossEncoder(model_name)
    scores = reranker.predict([[question, doc] for doc in docs])
    ranked = sorted(
        zip(ids, docs, metas, scores), key=lambda x: float(x[3]), reverse=True
    )[:top_k]
    return [x[0] for x in ranked], [x[1] for x in ranked], [x[2] for x in ranked]


def retrieve(question: str, settings: Settings | None = None):
    """Hybrid + rerank. Shared by ask() and eval tests."""
    settings = settings or get_settings()
    collection = get_collection(settings)
    ids, docs, metas = retrieve_hybrid(collection, question, settings.candidate_n)
    return rerank(
        question, ids, docs, metas, settings.cross_encoder_model, settings.top_k
    )


def _answer_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    parts = []
    for item in getattr(response, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip() or str(response.output)


def ask(question: str, settings: Settings | None = None) -> tuple[str, list[dict]]:
    settings = settings or get_settings()
    if not (
        settings.azure_openai_endpoint
        and settings.azure_openai_api_key
        and settings.azure_openai_deployment
    ):
        raise RuntimeError(
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT"
        )

    _ids, docs, metas = retrieve(question, settings)

    context = "\n\n".join(
        f"[{i+1}] ({m['document_name']} § {m.get('section', 'body')})\n{d}"
        for i, (d, m) in enumerate(zip(docs, metas))
    )
    prompt = (
        "Answer using only the context. If the context conflicts, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    client = OpenAI(
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
    )
    response = client.responses.create(
        model=settings.azure_openai_deployment,
        input=prompt,
    )
    return _answer_text(response), metas