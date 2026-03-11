# Inference: DCD and naive RAG

**What it does:** Reads questions from the pipeline dataset (`output/dataset.xlsx`), runs two inference flows, and writes results to Excel. **DCD:** classifies each question to domain and collection (LLM), queries Chroma with that metadata filter, reranks chunks with a cross-encoder, then generates an answer from the top context (LLM). **Naive RAG:** queries Chroma over the whole DB (no filter), takes top N chunks, generates an answer. Both use the same answer-generation LLM.

**Input:** `output/dataset.xlsx` (from “Generate dataset” stage) and the Chroma DB from “Create vector DB” stage.

**Output (in `output/` by default):**
- `dcd_dataset.xlsx` — original dataset columns + `find_context`, `find_domain`, `find_collection`, `generate_answer` (DCD results).
- `naive_rag_dataset.xlsx` — original dataset columns + `find_context`, `find_domain`, `find_collection`, `generate_answer` (Naive RAG results).

---

## Configuration

### 1. Environment (`.env`)

Set your LLM API and model (used for classification and answer generation):

| Variable    | Description                    |
|------------|--------------------------------|
| `API_KEY`  | API key for the LLM provider   |
| `BASE_URL` | Base URL of the API            |
| `MODEL_NAME` | Model name (e.g. `qwen2.5-7b-instruct`) |

### 2. `inference_config.yaml`

In this folder. Main options:

| Key | Description |
|-----|-------------|
| `pipeline_output_dir` | Folder (relative to parent of this script) where `dataset.xlsx` lives (default: `output`) |
| `dataset_file` | Dataset filename (default: `dataset.xlsx`) |
| `output_dir` | Folder for result files (relative to this folder) |
| `output_dcd_file`, `output_naive_rag_file` | Output Excel filenames |
| `chroma_path` | Path to Chroma DB directory (relative to this folder; e.g. `../3. Create vector DB/my_db`) |
| `collection_name` | Chroma collection name |
| `reranker_model` | Cross-encoder model for reranking (e.g. `BAAI/bge-reranker-v2-m3`) |
| `dcd_n_results` | Chroma query size before rerank |
| `dcd_rerank_top_n` | Number of chunks kept after rerank |
| `naive_rag_n_results` | Number of chunks for naive RAG |
| `domain_display_to_key`, `collection_display_to_key` | Display name → key maps for classifier; must match dataset/metadata |

---

## Run

From this folder:

```bash
pip install pandas openai chromadb sentence-transformers pyyaml python-dotenv pydantic openpyxl
python main.py
```

The script loads the dataset, runs DCD then naive RAG for every row, and appends results to the two Excel files (overwrites per run). Paths are relative to the script directory and its parent as in the config.
