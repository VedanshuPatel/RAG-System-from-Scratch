"""CLI for the RAG ask pipeline."""

import sys

from rag.pipeline import ask


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit('Usage: python scripts/ask.py "your question"')

    answer, sources = ask(question)
    print("Answer:")
    print(answer)
    print("\nSources:")
    for m in sources:
        print(f"- {m['document_name']} § {m.get('section', 'body')}")


if __name__ == "__main__":
    main()