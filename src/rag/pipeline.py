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
    by_id = {
        chunk_id: (doc, meta)
        for chunk_id, doc, meta in zip(vec_ids, vec_docs, vec_metas)
    }

    all_rows = collection.get(include=["documents", "metadatas"])
    question_lower = question.lower()
    ordered_ids = []
    for chunk_id, doc, meta in zip(
        all_rows["ids"], all_rows["documents"], all_rows["metadatas"]
    ):
        if question_lower in doc.lower():
            by_id[chunk_id] = (doc, meta)
            if chunk_id not in ordered_ids:
                ordered_ids.append(chunk_id)
    for chunk_id in vec_ids:
        if chunk_id not in ordered_ids:
            ordered_ids.append(chunk_id)

    return (
        ordered_ids,
        [by_id[chunk_id][0] for chunk_id in ordered_ids],
        [by_id[chunk_id][1] for chunk_id in ordered_ids],
    )


def rerank(question, ids, docs, metas, model_name: str, top_k: int):
    reranker = CrossEncoder(model_name)
    scores = reranker.predict([[question, doc] for doc in docs])
    ranked = sorted(
        zip(ids, docs, metas, scores),
        key=lambda row: float(row[3]),
        reverse=True,
    )[:top_k]
    return (
        [row[0] for row in ranked],
        [row[1] for row in ranked],
        [row[2] for row in ranked],
    )


def retrieve(question: str, settings: Settings | None = None):
    """Hybrid + rerank. Shared by ask() and eval tests."""
    settings = settings or get_settings()
    collection = get_collection(settings)
    ids, docs, metas = retrieve_hybrid(collection, question, settings.candidate_n)
    return rerank(
        question, ids, docs, metas, settings.cross_encoder_model, settings.top_k
    )


def _require_azure(settings: Settings) -> None:
    if not (
        settings.azure_openai_endpoint
        and settings.azure_openai_api_key
        and settings.azure_openai_deployment
    ):
        raise RuntimeError(
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT"
        )


def _build_context(docs: list[str], metas: list[dict]) -> str:
    return "\n\n".join(
        f"[{index + 1}] ({meta['document_name']} § {meta.get('section', 'body')})\n{doc}"
        for index, (doc, meta) in enumerate(zip(docs, metas))
    )


def _build_prompt(question: str, context: str) -> str:
    return (
        "Answer using only the context. If the context conflicts, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def _generate(prompt: str, settings: Settings) -> str:
    client = OpenAI(
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
    )
    response = client.responses.create(
        model=settings.azure_openai_deployment,
        input=prompt,
    )
    return _answer_text(response)


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
    _require_azure(settings)

    _ids, docs, metas = retrieve(question, settings)
    prompt = _build_prompt(question, _build_context(docs, metas))
    return _generate(prompt, settings), metas
