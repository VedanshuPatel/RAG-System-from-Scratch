from rag.chunking.structural import structural_chunk


SAMPLE = """# Remote Work Policy

## Eligibility
Employees with 6 months tenure may request remote work.

## Equipment
The company provides a laptop and stipend of $75/month.

## Security
VPN is required when accessing internal systems remotely.
"""


def test_structural_splits_on_headings():
    chunks = structural_chunk(SAMPLE, document_name="remote_work.md", chunk_size=500, overlap=50)
    assert len(chunks) >= 2
    assert all(c.document_name == "remote_work.md" for c in chunks)
    assert any("Eligibility" in (c.section or "") or "Eligibility" in c.text for c in chunks)


def test_structural_assigns_ids():
    chunks = structural_chunk(SAMPLE, document_name="remote_work.md", chunk_size=500, overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert all(ids)
    assert len(ids) == len(set(ids))


def test_structural_preserves_content():
    chunks = structural_chunk(SAMPLE, document_name="remote_work.md", chunk_size=2000, overlap=0)
    joined = " ".join(c.text for c in chunks)
    assert "VPN is required" in joined
    assert "$75/month" in joined