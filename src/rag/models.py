"""Shared data models for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A text span with source attribution metadata."""

    text: str
    document_name: str
    section: str = ""
    chunk_id: str = ""
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_id(self, chunk_id: str) -> Chunk:
        self.chunk_id = chunk_id
        return self

    def attribution(self) -> str:
        section = self.section or "body"
        return f"{self.document_name} § {section}"