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

def test_structural_no_headings_uses_body():
    chunks = structural_chunk(
        "Plain policy text with no markdown headings.",
        document_name="plain.md",
        chunk_size=500,
        overlap=50,
    )
    assert len(chunks) == 1
    assert chunks[0].section == "body"


def test_structural_keeps_preamble():
    text = "Intro before heading.\n\n# Title\n\nBody under title.\n"
    chunks = structural_chunk(text, document_name="p.md", chunk_size=500, overlap=20)
    assert any(c.section == "preamble" for c in chunks)
    assert any("Intro before heading" in c.text for c in chunks)


def test_structural_splits_oversized_section():
    body = "A" * 250
    text = f"# Big\n\n{body}\n"
    chunks = structural_chunk(text, document_name="big.md", chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 100 for c in chunks)


def test_structural_skips_empty_parts():
    text = "# Empty\n\n   \n"
    chunks = structural_chunk(text, document_name="e.md", chunk_size=50, overlap=5)
    # heading-only / whitespace body should not invent empty chunks beyond real content
    assert all(c.text.strip() for c in chunks)

def test_structural_skips_whitespace_only_body():
    # no headings; strip() yields empty body → _split_oversized returns [""] → continue
    chunks = structural_chunk("   \n\t  ", document_name="blank.md", chunk_size=50, overlap=5)
    assert chunks == []