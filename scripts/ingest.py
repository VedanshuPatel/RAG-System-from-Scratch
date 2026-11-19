"""Chunk policy docs and store embeddings in Chroma."""


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.chunking import get_chunker
from rag.config import get_settings


def main() -> None:
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))

    try:
        client.delete_collection(settings.collection_name)
    except Exception:  # noqa: BLE001, S110
        pass

    collection = client.create_collection(
        name=settings.collection_name,
        embedding_function=embed_fn,
    )

    chunker = get_chunker(settings.chunking_strategy)
    files = sorted(settings.data_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files in {settings.data_dir}")

    ids, docs, metas = [], [], []
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks = chunker(
            text,
            document_name=path.name,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        for c in chunks:
            ids.append(c.chunk_id)
            docs.append(c.text)
            metas.append(
                {
                    "document_name": c.document_name,
                    "section": c.section or "body",
                    "strategy": settings.chunking_strategy,
                }
            )

    collection.add(ids=ids, documents=docs, metadatas=metas)
    print(f"strategy={settings.chunking_strategy}")
    print(f"files={len(files)} chunks={len(ids)} -> {settings.chroma_dir}")


if __name__ == "__main__":
    main()