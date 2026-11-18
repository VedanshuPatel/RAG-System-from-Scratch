import pytest

from rag.chunking import get_chunker
from rag.config import VALID_STRATEGIES


def test_factory_returns_callable_for_each_strategy():
    for name in VALID_STRATEGIES:
        chunker = get_chunker(name)
        assert callable(chunker)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        get_chunker("bogus")


def test_only_one_strategy_used_from_settings(monkeypatch):
    monkeypatch.setenv("CHUNKING_STRATEGY", "structural")
    from rag.config import get_settings

    settings = get_settings()
    assert settings.chunking_strategy == "structural"
    chunker = get_chunker(settings.chunking_strategy)
    chunks = chunker("# A\n\nHello world.\n", document_name="a.md", chunk_size=100, overlap=10)
    assert isinstance(chunks, list)