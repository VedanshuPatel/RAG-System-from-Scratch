"""Late chunking spans: character windows with offsets for later token pooling."""

from __future__ import annotations

from rag.models import Chunk


def late_chunk_spans(
    text: str,
    *,
    document_name: str,
    chunk_size: int = 1800,
    overlap: int = 200,
) -> list[Chunk]:
    if not text:
        return []

    spans: list[Chunk] = []
    start = 0
    idx = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        # Prefer breaking on whitespace when not at the end
        if end < n:
            pivot = text.rfind(" ", start, end)
            if pivot > start:
                end = pivot
        piece = text[start:end]
        spans.append(
            Chunk(
                text=piece,
                document_name=document_name,
                section="span",
                chunk_id=f"{document_name}::late::{idx}",
                start_char=start,
                end_char=end,
                metadata={"strategy": "late"},
            )
        )
        idx += 1
        if end >= n:
            break
        next_start = max(start + 1, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    if spans and spans[-1].end_char != n:
        # Ensure coverage to EOF if trailing whitespace edge-case
        last = spans[-1]
        if last.end_char < n:
            spans.append(
                Chunk(
                    text=text[last.end_char : n],
                    document_name=document_name,
                    section="span",
                    chunk_id=f"{document_name}::late::{idx}",
                    start_char=last.end_char,
                    end_char=n,
                    metadata={"strategy": "late"},
                )
            )
    return spans