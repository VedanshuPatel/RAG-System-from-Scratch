"""Context-aware chunking: structural splits + document/section prefixes."""

from __future__ import annotations

from rag.chunking.structural import structural_chunk
from rag.models import Chunk


def context_aware_chunk(
    text: str,
    *,
    document_name: str,
    chunk_size: int = 1800,
    overlap: int = 200,
) -> list[Chunk]:
    base = structural_chunk(
        text, document_name=document_name, chunk_size=chunk_size, overlap=overlap
    )
    enriched: list[Chunk] = []
    for i, chunk in enumerate(base):
        section = chunk.section or "body"
        prefixed = (
            f"Document: {document_name}\n"
            f"Section: {section}\n\n"
            f"{chunk.text}"
        )
        enriched.append(
            Chunk(
                text=prefixed,
                document_name=document_name,
                section=section,
                chunk_id=f"{document_name}::context_aware::{i}",
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={"strategy": "context_aware"},
            )
        )
    return enriched