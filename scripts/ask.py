"""CLI for the RAG ask pipeline."""

import sys

from rag.pipeline import ask


def main() -> None:
    """Ask one question from the CLI and print the answer with sources."""
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python scripts/ask.py "your question"')

    answer, sources = ask(question)
    print("Answer:")
    print(answer)
    print("\nSources:")
    for meta in sources:
        print(f"- {meta['document_name']} § {meta.get('section', 'body')}")


if __name__ == "__main__":
    main()