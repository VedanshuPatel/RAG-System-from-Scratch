"""Print retrieval recall and answer accuracy for the eval set."""

import os

from eval_cases import CASES
from rag.pipeline import ask, retrieve


def main() -> None:
    recall_hits = 0
    print("=== Retrieval recall (doc + chunk id) ===")
    for case in CASES:
        ids, _docs, metas = retrieve(case["question"])
        docs_hit = {m.get("document_name") for m in metas}
        doc_ok = case["expected_doc"] in docs_hit
        chunk_ok = bool(set(case["expected_chunk_ids"]) & set(ids))
        ok = doc_ok and chunk_ok
        recall_hits += int(ok)
        print(
            f"{'PASS' if ok else 'FAIL'} {case['id']}: "
            f"doc={doc_ok} chunk={chunk_ok} top={ids[:3]}"
        )

    recall = recall_hits / len(CASES)
    print(f"\nRetrieval recall: {recall_hits}/{len(CASES)} = {recall:.3f}")

    if os.getenv("RUN_LIVE", "0") != "1":
        print("\nSkip generator accuracy (set RUN_LIVE=1 to include).")
        return

    answer_hits = 0
    print("\n=== Generator answer accuracy (key facts) ===")
    for case in CASES:
        answer, _sources = ask(case["question"])
        lower = answer.lower()
        ok = True
        for k in case["expected_answer_keys"]:
            alts = [a.strip().lower() for a in k.split("|")]
            if not any(a in lower for a in alts):
                ok = False
                break
        answer_hits += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} {case['id']}: keys={case['expected_answer_keys']}")

    accuracy = answer_hits / len(CASES)
    print(f"\nAnswer accuracy: {answer_hits}/{len(CASES)} = {accuracy:.3f}")


if __name__ == "__main__":
    main()