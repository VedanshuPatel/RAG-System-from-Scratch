import os

import pytest

from rag.pipeline import ask, retrieve
from eval_cases import CASES


def _keys_present(text: str, keys: list[str]) -> bool:
    lower = text.lower()
    return all(k.lower() in lower for k in keys)


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_retriever_recall(case):
    ids, _docs, metas = retrieve(case["question"])
    assert ids, f"No results for {case['id']}"

    docs_hit = {m.get("document_name") for m in metas}
    assert case["expected_doc"] in docs_hit, (
        f"[{case['id']}] expected doc {case['expected_doc']} not in {sorted(docs_hit)}"
    )

    expected_chunks = set(case["expected_chunk_ids"])
    assert expected_chunks & set(ids), (
        f"[{case['id']}] expected one of {case['expected_chunk_ids']} in top ids {ids}"
    )


@pytest.mark.live
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_generator_answer_keys(case):
    if os.getenv("RUN_LIVE", "0") != "1":
        pytest.skip("Set RUN_LIVE=1 to run generator tests")

    answer, sources = ask(case["question"])
    assert answer.strip(), f"[{case['id']}] empty answer"
    assert _keys_present(answer, case["expected_answer_keys"]), (
        f"[{case['id']}] missing {case['expected_answer_keys']} in:\n{answer}"
    )
    assert sources, f"[{case['id']}] no sources"