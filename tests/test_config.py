import pytest

from conftest import make_settings


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="CHUNKING_STRATEGY"):
        make_settings(chunking_strategy="late")


def test_overlap_too_large_raises():
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        make_settings(chunk_size=100, chunk_overlap=100)