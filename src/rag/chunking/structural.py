"""Structural chunking: split on markdown headings, then size windows with overlap."""

from __future__ import annotations

import re

from rag.models import Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _split_oversized(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        parts.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return parts


def structural_chunk(
    text: str,
    *,
    document_name: str,
    chunk_size: int = 1800,
    overlap: int = 200,
) -> list[Chunk]:
    matches = list(_HEADING.finditer(text))
    sections: list[tuple[str, str]] = []

    if not matches:
        sections.append(("body", text.strip()))
    else:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("preamble", preamble))
        for i, match in enumerate(matches):
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            section_text = f"{match.group(0).strip()}\n{body}".strip()
            sections.append((title, section_text))

    chunks: list[Chunk] = []
    idx = 0
    for section, section_text in sections:
        for part in _split_oversized(section_text, chunk_size, overlap):
            if not part.strip():
                continue
            chunk_id = f"{document_name}::structural::{idx}"
            chunks.append(
                Chunk(
                    text=part,
                    document_name=document_name,
                    section=section,
                    chunk_id=chunk_id,
                    start_char=0,
                    end_char=len(part),
                    metadata={"strategy": "structural"},
                )
            )
            idx += 1
    return chunks