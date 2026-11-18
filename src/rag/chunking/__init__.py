"""Chunking strategies. Only one is selected at runtime via get_chunker()."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rag.chunking.context_aware import context_aware_chunk
from rag.chunking.late import late_chunk_spans
from rag.chunking.structural import structural_chunk
from rag.config import VALID_STRATEGIES

Chunker = Callable[..., list[Any]]

_REGISTRY: dict[str, Chunker] = {
    "structural": structural_chunk,
    "late": late_chunk_spans,
    "context_aware": context_aware_chunk,
}


def get_chunker(strategy: str) -> Chunker:
    key = strategy.strip().lower()
    if key not in VALID_STRATEGIES or key not in _REGISTRY:
        raise ValueError(
            f"Unknown chunking strategy {strategy!r}. "
            f"Choose one of: {sorted(VALID_STRATEGIES)}"
        )
    return _REGISTRY[key]