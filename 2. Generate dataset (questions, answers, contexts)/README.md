# Generate dataset (Q&A, contexts)

**What it does:** Reads existing `.txt` documents from `../output/<domain>/<collection>/`, calls an LLM to generate N question–answer pairs per document (with exact context snippets), and writes the RAG dataset. Also builds `metadata_mapping.json` for domain/collection display names used in question prefixes and filtering.

**Input:** Pre-generated documents under `../output/` (from the “Create text dataset” stage).

**Output (in `../output/`):**
- `dataset.xlsx` — questions with prefix «RC X», section «Y», answers, contexts
- `dataset_classification.json` — full records: question, answer, context, domain, collection, document
- `metadata_mapping.json` — domain_key → display name, collection_key → display name (for RAG metadata filter)

---

## Configuration

### 1. Environment (`.env`)

Copy `.env.example` to `.env` and set:

| Variable | Description |
|----------|-------------|
| `OPENAI_BASE_URL` | API base URL (default: `https://api.openai.com/v1`) |
| `OPENAI_API_KEY` | API key (required) |
| `OPENAI_MODEL` | Model name (e.g. `gpt-4o`) |

### 2. `dataset_config.yaml`

In this folder. Options:

| Key | Description |
|-----|-------------|
| `qa_pairs_per_document` | Number of Q&A pairs to generate per document (default: 2) |
| `domain_display_names` | Map `domain_key` → human-readable name (used in question prefix and metadata) |
| `collection_display_names` | Map `collection_key` → human-readable name |

Add or edit keys to match your domains/collections from the text dataset stage.

---

## Run

From this folder:

```bash
pip install pyyaml python-dotenv openpyxl  # if not already installed
python generate_dataset.py
```

The script clears existing `dataset.xlsx` and `dataset_classification.json`, then processes every `.txt` under `../output/` and appends rows. Logs progress per document.
