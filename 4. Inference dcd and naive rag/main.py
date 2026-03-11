"""Run DCD and naive RAG inference on dataset questions; save results to Excel. Config: inference_config.yaml."""

from pathlib import Path

import pandas as pd

from config import get_inference_config
from dcd_pipe import pipeline_dcd
from naive_rag import pipeline_naive_rag
from utils import create_classification_schema, generate_answer

from dotenv import load_dotenv

load_dotenv()

_SCRIPT_DIR = Path(__file__).resolve().parent
_cfg = get_inference_config()

# Dataset: pipeline output folder (parent of script dir) / pipeline_output_dir / dataset_file
_pipeline_output = _SCRIPT_DIR.parent / _cfg.get("pipeline_output_dir", "output")
_dataset_file = _cfg.get("dataset_file", "dataset.xlsx")
DATASET_PATH = _pipeline_output / _dataset_file

# Output paths (script dir / output_dir / filename)
_output_dir = _cfg.get("output_dir", "output")
OUTPUT_PATH_DCD = _SCRIPT_DIR / _output_dir / _cfg.get("output_dcd_file", "dcd_dataset.xlsx")
OUTPUT_PATH_NAIVE_RAG = _SCRIPT_DIR / _output_dir / _cfg.get("output_naive_rag_file", "naive_rag_dataset.xlsx")

# Domain/collection display names for classification schema (order from config)
_domain_to_key = _cfg.get("domain_display_to_key", {})
_collection_to_key = _cfg.get("collection_display_to_key", {})
domains = list(_domain_to_key.keys())
collections = list(_collection_to_key.keys())

NO_CONTEXT_MESSAGE = "[No context found for this query]"


df = pd.read_excel(DATASET_PATH)
domains_scheme, collection_scheme = create_classification_schema(domains, collections)

# DCD inference
print("\n" + "="*60)
print("Running DCD inference...")
print("="*60)

df_dcd = df.copy()
find_contexts_dcd = []
find_domains_dcd = []
find_collections_dcd = []
generate_answers_dcd = []

for index, row in df_dcd.iterrows():
    print(f"{index + 1}/{len(df_dcd)}: DCD")
    
    question = row["question"]
    search_results = pipeline_dcd(question, domains_scheme, collection_scheme)

    if not search_results:
        find_context = ""
        find_metadata = {"domain": "", "collection": ""}
        answer_question = NO_CONTEXT_MESSAGE
    else:
        combined_context = "\n---\n".join([res["content"] for res in search_results])
        find_context = combined_context
        find_metadata = search_results[0]["metadata"]
        answer_question = generate_answer(question, combined_context)
    
    find_contexts_dcd.append(find_context)
    find_domains_dcd.append(find_metadata.get("domain", ""))
    find_collections_dcd.append(find_metadata.get("collection", ""))
    generate_answers_dcd.append(answer_question)

df_dcd["find_context"] = find_contexts_dcd
df_dcd["find_domain"] = find_domains_dcd
df_dcd["find_collection"] = find_collections_dcd
df_dcd["generate_answer"] = generate_answers_dcd

OUTPUT_PATH_DCD.parent.mkdir(parents=True, exist_ok=True)
df_dcd.to_excel(OUTPUT_PATH_DCD, index=False)
print(f"\nDCD results saved → {OUTPUT_PATH_DCD}")

# Naive RAG inference
print("\n" + "="*60)
print("Running Naive RAG inference...")
print("="*60)

df_naive = df.copy()
find_contexts_naive = []
find_domains_naive = []
find_collections_naive = []
generate_answers_naive = []

for index, row in df_naive.iterrows():
    print(f"{index + 1}/{len(df_naive)}: Naive RAG")
    
    question = row["question"]
    search_results = pipeline_naive_rag(question)

    if not search_results:
        find_context = ""
        find_metadata = {"domain": "", "collection": ""}
        answer_question = NO_CONTEXT_MESSAGE
    else:
        combined_context = "\n---\n".join([res["content"] for res in search_results])
        find_context = combined_context
        find_metadata = search_results[0]["metadata"]
        answer_question = generate_answer(question, combined_context)
    
    find_contexts_naive.append(find_context)
    find_domains_naive.append(find_metadata.get("domain", ""))
    find_collections_naive.append(find_metadata.get("collection", ""))
    generate_answers_naive.append(answer_question)

df_naive["find_context"] = find_contexts_naive
df_naive["find_domain"] = find_domains_naive
df_naive["find_collection"] = find_collections_naive
df_naive["generate_answer"] = generate_answers_naive

df_naive.to_excel(OUTPUT_PATH_NAIVE_RAG, index=False)
print(f"\nNaive RAG results saved → {OUTPUT_PATH_NAIVE_RAG}")

print("\n" + "="*60)
print("Inference complete!")
print("="*60)
