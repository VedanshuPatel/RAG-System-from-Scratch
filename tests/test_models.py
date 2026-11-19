from rag.models import Chunk


def test_with_id_sets_and_returns_self():
    c = Chunk(text="hi", document_name="a.md")
    out = c.with_id("a.md::structural::0")
    assert out is c
    assert c.chunk_id == "a.md::structural::0"


def test_attribution_with_section():
    c = Chunk(text="x", document_name="pto.md", section="Accrual")
    assert c.attribution() == "pto.md § Accrual"


def test_attribution_defaults_to_body():
    c = Chunk(text="x", document_name="pto.md", section="")
    assert c.attribution() == "pto.md § body"