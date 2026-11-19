"""Ask a question over ingested policy chunks (vector retrieve + Azure LLM)."""

import sys

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI

from rag.config import get_settings


def answer_text(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    parts = []
    for item in getattr(response, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip() or str(response.output)


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python scripts/ask.py "your question"')

    settings = get_settings()
    if not (
        settings.azure_openai_endpoint
        and settings.azure_openai_api_key
        and settings.azure_openai_deployment
    ):
        raise SystemExit("Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT in .env")

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    collection = chromadb.PersistentClient(path=str(settings.chroma_dir)).get_collection(
        name=settings.collection_name,
        embedding_function=embed_fn,
    )

    result = collection.query(query_texts=[question], n_results=settings.top_k)
    docs = result["documents"][0]
    metas = result["metadatas"][0]

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

    print("Answer:")
    print(answer_text(response))
    print("\nSources:")
    for m in metas:
        print(f"- {m['document_name']} § {m.get('section', 'body')}")


if __name__ == "__main__":
    main()