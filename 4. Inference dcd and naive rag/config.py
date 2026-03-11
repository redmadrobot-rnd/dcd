"""Load inference_config.yaml and .env; expose Chroma client, OpenAI client, reranker, and display-to-key mappings."""

import os
from pathlib import Path

import chromadb
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

load_dotenv()

_CONFIG_PATH = Path(__file__).parent / "inference_config.yaml"
_inference_config: dict | None = None


def get_inference_config() -> dict:
    """Load inference_config.yaml (cached)."""
    global _inference_config
    if _inference_config is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _inference_config = yaml.safe_load(f) or {}
    return _inference_config


def _resolve_chroma_path() -> Path:
    cfg = get_inference_config()
    chroma_path = cfg.get("chroma_path", "my_db")
    base = Path(__file__).resolve().parent
    return (base / chroma_path).resolve()


# Build from config
_cfg = get_inference_config()
DOMAIN_DISPLAY_TO_KEY = _cfg.get("domain_display_to_key", {})
COLLECTION_DISPLAY_TO_KEY = _cfg.get("collection_display_to_key", {})

client = chromadb.PersistentClient(path=str(_resolve_chroma_path()))
collection_chroma = client.get_collection(_cfg.get("collection_name", "dcd_collection"))

reranker_model_name = _cfg.get("reranker_model", "BAAI/bge-reranker-v2-m3")
reranker_model = CrossEncoder(reranker_model_name)

client_openai = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

DCD_N_RESULTS = _cfg.get("dcd_n_results", 20)
DCD_RERANK_TOP_N = _cfg.get("dcd_rerank_top_n", 10)
NAIVE_RAG_N_RESULTS = _cfg.get("naive_rag_n_results", 10)
