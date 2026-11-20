from rag.models import Chunk


def test_with_id_sets_and_returns_self():
    chunk = Chunk(text="hi", document_name="a.md")
    out = chunk.with_id("a.md::structural::0")
    assert out is chunk
    assert chunk.chunk_id == "a.md::structural::0"


def test_attribution_with_section():
    chunk = Chunk(text="x", document_name="pto.md", section="Accrual")
    assert chunk.attribution() == "pto.md § Accrual"


def test_attribution_defaults_to_body():
    chunk = Chunk(text="x", document_name="pto.md", section="")
    assert chunk.attribution() == "pto.md § body"
