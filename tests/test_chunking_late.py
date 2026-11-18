from rag.chunking.late import late_chunk_spans


SAMPLE = """# Expense Policy

## Eligible Expenses
Travel, meals, and lodging are reimbursable with receipts.

## Limits
Meals are capped at $50 per day without manager approval.
"""


def test_late_returns_spans_with_offsets():
    spans = late_chunk_spans(SAMPLE, document_name="expense.md", chunk_size=200, overlap=20)
    assert spans
    for c in spans:
        assert c.start_char >= 0
        assert c.end_char > c.start_char
        assert SAMPLE[c.start_char : c.end_char]
        assert c.document_name == "expense.md"


def test_late_spans_cover_document():
    spans = late_chunk_spans(SAMPLE, document_name="expense.md", chunk_size=120, overlap=10)
    assert spans[0].start_char == 0
    assert spans[-1].end_char == len(SAMPLE)