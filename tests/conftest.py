from rag.config import Settings


def make_settings(**kwargs) -> Settings:
    base = {
        "chunking_strategy": "structural",
        "embedding_model": "embed-model",
        "cross_encoder_model": "rerank-model",
        "chunk_size": 1800,
        "chunk_overlap": 200,
        "top_k": 5,
        "candidate_n": 20,
        "data_dir": ".",
        "chroma_dir": ".",
        "collection_name": "policies",
        "azure_openai_endpoint": None,
        "azure_openai_api_key": None,
        "azure_openai_deployment": None,
        "run_live": False,
        "traces_dir": ".",
    }
    base.update(kwargs)
    return Settings(**base)