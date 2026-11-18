from rag.chunking.context_aware import context_aware_chunk


SAMPLE = """# PTO Policy

## Accrual
Full-time employees accrue 1.5 days per month.

## Carryover
Unused PTO may carry over up to 5 days.
"""


def test_context_aware_prefixes_document_and_section():
    chunks = context_aware_chunk(
        SAMPLE, document_name="pto.md", chunk_size=800, overlap=50
    )
    assert chunks
    assert any("Document: pto.md" in c.text for c in chunks)
    assert any("Section:" in c.text for c in chunks)


def test_context_aware_keeps_body():
    chunks = context_aware_chunk(
        SAMPLE, document_name="pto.md", chunk_size=800, overlap=50
    )
    joined = " ".join(c.text for c in chunks)
    assert "1.5 days" in joined