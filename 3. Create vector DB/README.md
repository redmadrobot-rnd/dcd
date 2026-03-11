# Create vector DB

**What it does:** Reads `.txt` documents from the output folder (`../output/<domain>/<collection>/`), splits them into chunks (by token count), embeds with a local Sentence Transformer model, and writes them into a Chroma persistent collection. Also saves a JSON dump of chunks for inspection.

**Input:** Documents under `../output/` (from the “Create text dataset” stage).

**Output:**
- Chroma DB in `./my_db` (path configurable)
- `chunks_output.json` — all chunks with metadata (domain, collection, document, chunk_number)

---

## Configuration

Edit `vector_db_config.yaml` in this folder:

| Key | Description |
|-----|-------------|
| `chroma_path` | Directory for Chroma persistent DB (default: `./my_db`) |
| `collection_name` | Chroma collection name |
| `embedding_model` | Sentence Transformer model (e.g. `BAAI/bge-m3`) |
| `tokenizer_encoding` | Tiktoken encoding for chunk length (e.g. `cl100k_base`) |
| `chunk_size` | Max tokens per chunk |
| `chunk_overlap` | Overlap between chunks |
| `separators` | Split order (paragraph, line, sentence, word) |
| `documents_root` | Path to documents (default: `../output`) |
| `chunks_output_file` | Output JSON file name |
| `batch_size` | Batch size when adding to Chroma |

---

## Run

From this folder:

```bash
pip install chromadb sentence-transformers pyyaml tiktoken langchain-text-splitters tqdm
python prepare_vector_db.py
```

Paths are resolved relative to the script directory. The script creates or reuses the Chroma collection and appends chunks in batches.
