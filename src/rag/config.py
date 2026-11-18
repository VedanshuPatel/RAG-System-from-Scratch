"""Central configuration. Chunk size/overlap rationale documented below."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ~1800 chars ≈ 400–500 tokens for policy prose.
# ~200 char overlap (~10–15%) keeps eligibility/amount sentences from
# splitting cleanly across adjacent chunk boundaries.
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 200
VALID_STRATEGIES = frozenset({"structural"})

@dataclass(frozen=True)
class Settings:
    chunking_strategy: str
    embedding_model: str
    cross_encoder_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    candidate_n: int
    data_dir: Path
    chroma_dir: Path
    collection_name: str
    azure_openai_endpoint: str | None
    azure_openai_api_key: str | None
    azure_openai_deployment: str | None
    run_live: bool
    traces_dir: Path

    def __post_init__(self) -> None:
        if self.chunking_strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"CHUNKING_STRATEGY must be one of {sorted(VALID_STRATEGIES)}, "
                f"got {self.chunking_strategy!r}"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    strategy = os.getenv("CHUNKING_STRATEGY", "structural").strip().lower()
    data_dir = Path(os.getenv("DATA_DIR", "data/policies"))
    chroma_dir = Path(os.getenv("CHROMA_DIR", "data/chroma"))
    traces_dir = Path(os.getenv("TRACES_DIR", "data/traces"))
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    if not chroma_dir.is_absolute():
        chroma_dir = root / chroma_dir
    if not traces_dir.is_absolute():
        traces_dir = root / traces_dir

    return Settings(
        chunking_strategy=strategy,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        cross_encoder_model=os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))),
        top_k=int(os.getenv("TOP_K", "5")),
        candidate_n=int(os.getenv("CANDIDATE_N", "20")),
        data_dir=data_dir,
        chroma_dir=chroma_dir,
        collection_name=os.getenv("COLLECTION_NAME", "policies"),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        run_live=os.getenv("RUN_LIVE", "0") == "1",
        traces_dir=traces_dir,
    )