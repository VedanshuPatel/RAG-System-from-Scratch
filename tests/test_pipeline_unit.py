from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from conftest import make_settings

from rag import pipeline


class FakeCollection:
    def __init__(self, rows, vec_ids):
        self.rows = rows
        self.vec_ids = vec_ids

    def query(self, query_texts, n_results):
        by_id = {
            chunk_id: (doc, meta) for chunk_id, doc, meta in self.rows
        }
        ids = [chunk_id for chunk_id in self.vec_ids if chunk_id in by_id][:n_results]
        return {
            "ids": [ids],
            "documents": [[by_id[chunk_id][0] for chunk_id in ids]],
            "metadatas": [[by_id[chunk_id][1] for chunk_id in ids]],
        }

    def get(self, include=None):
        return {
            "ids": [row[0] for row in self.rows],
            "documents": [row[1] for row in self.rows],
            "metadatas": [row[2] for row in self.rows],
        }


def test_retrieve_hybrid_merges_keyword_then_vector():
    rows = [
        ("a", "alpha meal limit fifty", {"document_name": "a.md"}),
        ("b", "beta unrelated", {"document_name": "b.md"}),
        ("c", "gamma meal keyword hit", {"document_name": "c.md"}),
    ]
    coll = FakeCollection(rows, vec_ids=["b", "a"])
    ids, docs, metas = pipeline.retrieve_hybrid(coll, "meal", candidate_n=10)
    assert ids.index("a") < ids.index("b")
    assert ids.index("c") < ids.index("b")
    assert "a" in ids and "b" in ids and "c" in ids
    assert len(ids) == len(set(ids))
    assert len(docs) == len(ids) == len(metas)


def test_rerank_orders_by_score(monkeypatch):
    class FakeCE:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs):
            return [2.0 if "best" in pair[1] else 0.1 for pair in pairs]

    monkeypatch.setattr(pipeline, "CrossEncoder", FakeCE)
    ids, docs, _metas = pipeline.rerank(
        "q",
        ["1", "2"],
        ["worst text", "best text"],
        [{"document_name": "a.md"}, {"document_name": "b.md"}],
        model_name="ce",
        top_k=1,
    )
    assert ids == ["2"]
    assert docs == ["best text"]


def test_answer_text_prefers_output_text():
    resp = SimpleNamespace(output_text="hello", output=[])
    assert pipeline._answer_text(resp) == "hello"


def test_answer_text_falls_back_to_blocks():
    block = SimpleNamespace(text="from block")
    item = SimpleNamespace(content=[block])
    resp = SimpleNamespace(output_text="", output=[item])
    assert pipeline._answer_text(resp) == "from block"


def test_answer_text_stringifies_output_when_empty():
    resp = SimpleNamespace(output_text="", output="raw")
    assert pipeline._answer_text(resp) == "raw"


def test_ask_requires_azure_settings():
    settings = make_settings(
        azure_openai_endpoint=None,
        azure_openai_api_key=None,
        azure_openai_deployment=None,
    )
    with pytest.raises(RuntimeError, match="AZURE_OPENAI"):
        pipeline.ask("q", settings=settings)


def test_ask_returns_answer_and_metas(monkeypatch):
    settings = make_settings(
        azure_openai_endpoint="https://example.openai.azure.com/openai/v1",
        azure_openai_api_key="key",
        azure_openai_deployment="gpt",
    )
    metas = [{"document_name": "a.md", "section": "Meals"}]

    monkeypatch.setattr(
        pipeline,
        "retrieve",
        lambda question, settings=None: (["id1"], ["context doc"], metas),
    )

    fake_client = MagicMock()
    fake_client.responses.create.return_value = SimpleNamespace(
        output_text="The limit is $50.",
        output=[],
    )
    monkeypatch.setattr(pipeline, "OpenAI", lambda **kwargs: fake_client)

    answer, sources = pipeline.ask("meal limit?", settings=settings)
    assert "$50" in answer
    assert sources == metas
    fake_client.responses.create.assert_called_once()


def test_retrieve_wires_hybrid_and_rerank(monkeypatch):
    settings = make_settings(top_k=1, candidate_n=5)
    monkeypatch.setattr(pipeline, "get_collection", lambda settings=None: "coll")
    monkeypatch.setattr(
        pipeline,
        "retrieve_hybrid",
        lambda collection, question, candidate_n: (
            ["1", "2"],
            ["d1", "d2"],
            [{"document_name": "a.md"}, {"document_name": "b.md"}],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "rerank",
        lambda question, ids, docs, metas, model_name, top_k: (
            ["2"],
            ["d2"],
            [{"document_name": "b.md"}],
        ),
    )
    ids, docs, _metas = pipeline.retrieve("q", settings=settings)
    assert ids == ["2"]
    assert docs == ["d2"]


def test_get_collection_uses_chroma(monkeypatch):
    settings = make_settings(collection_name="policies", chroma_dir="/tmp/chroma-test")
    fake_collection = object()
    fake_client = MagicMock()
    fake_client.get_collection.return_value = fake_collection

    monkeypatch.setattr(
        pipeline,
        "SentenceTransformerEmbeddingFunction",
        lambda model_name: "embed-fn",
    )
    monkeypatch.setattr(
        pipeline.chromadb,
        "PersistentClient",
        lambda path: fake_client,
    )

    out = pipeline.get_collection(settings)
    assert out is fake_collection
    fake_client.get_collection.assert_called_once_with(
        name="policies",
        embedding_function="embed-fn",
    )


def test_get_collection_defaults_settings(monkeypatch):
    settings = make_settings()
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    fake_client = MagicMock()
    fake_client.get_collection.return_value = "coll"
    monkeypatch.setattr(
        pipeline, "SentenceTransformerEmbeddingFunction", lambda model_name: "fn"
    )
    monkeypatch.setattr(pipeline.chromadb, "PersistentClient", lambda path: fake_client)

    assert pipeline.get_collection() == "coll"