"""Chunk policy docs and store embeddings in Chroma."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.chunking import get_chunker
from rag.config import Settings, get_settings


def _reset_collection(settings: Settings):
    """Delete and recreate the Chroma collection (fresh ingest each run)."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))

    try:
        client.delete_collection(settings.collection_name)
    except Exception:  # noqa: BLE001, S110
        pass

    return client.create_collection(
        name=settings.collection_name,
        embedding_function=embed_fn,
    )


def _collect_chunks(
    files: list[Path], settings: Settings
) -> tuple[list[str], list[str], list[dict]]:
    """Chunk each markdown file and return parallel ids, texts, and metadatas."""
    chunker = get_chunker(settings.chunking_strategy)
    ids, docs, metas = [], [], []
    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks = chunker(
            text,
            document_name=path.name,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            docs.append(chunk.text)
            metas.append(
                {
                    "document_name": chunk.document_name,
                    "section": chunk.section or "body",
                    "strategy": settings.chunking_strategy,
                }
            )
    return ids, docs, metas


def main() -> None:
    """Ingest all ``*.md`` files under ``DATA_DIR`` into Chroma."""
    settings = get_settings()
    files = sorted(settings.data_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"No .md files in {settings.data_dir}")

    collection = _reset_collection(settings)
    ids, docs, metas = _collect_chunks(files, settings)
    collection.add(ids=ids, documents=docs, metadatas=metas)

    print(f"strategy={settings.chunking_strategy}")
    print(f"files={len(files)} chunks={len(ids)} -> {settings.chroma_dir}")


if __name__ == "__main__":
    main()
