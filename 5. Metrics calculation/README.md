# Metrics Calculation

Unified metrics calculation module for RAG evaluation.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure `metrics_config.yaml`:
   - Verify input dataset paths point to step 4 output files
   - Set OpenAI API credentials (`base_url`, `api_key`, `model_name`)
   - Adjust output paths if needed

## Usage

```bash
python main.py
```

The script will:
1. Load `naive_rag_dataset.xlsx` and calculate **SB ARC, SB CR, SB FA** + **Context Score** metrics for Naive RAG
2. Load `dcd_dataset.xlsx` and calculate **SB ARC, SB CR, SB FA** + **Context Score** metrics for DCD
3. Save detailed per-sample results to Excel files
4. Save 2 separate metric files: `results/naive_rag_metrics.txt` and `results/dcd_metrics.txt`

## Calculated Metrics

### Generation Metrics (both Naive RAG and DCD):
- **SB ARC** (Answer Relevance & Completeness): Binary score based on 4 criteria (Direct, Complete, Specific, No Vagueness)
- **SB CR** (Context Recall): Whether answer uses ALL relevant facts from context
- **SB FA** (Factual Accuracy): Whether answer is factually accurate and hallucination-free

### Context Retrieval Metric:
- **Context Score** (0-2 scale): Compares retrieved context quality vs. reference context
  - 0: Irrelevant/Useless
  - 1: Partial/Flawed (on-topic but missing details or factual mismatches)
  - 2: Perfect Match (all key facts accurate)

The metrics are calculated independently for each RAG approach to compare their performance.

## Output Files

- `results/naive_rag_metrics.txt` — Naive RAG metrics (SB ARC, SB CR, SB FA, Context Score)
- `results/dcd_metrics.txt` — DCD metrics (SB ARC, SB CR, SB FA, Context Score)
- `results/naive_rag_evaluated.xlsx` — Detailed per-sample Naive RAG evaluation with reasoning
- `results/dcd_evaluated.xlsx` — Detailed per-sample DCD evaluation with reasoning

## Configuration

All parameters are in `metrics_config.yaml`:

```yaml
input:
  naive_rag_dataset: "path/to/naive_rag_dataset.xlsx"
  dcd_dataset: "path/to/dcd_dataset.xlsx"

output:
  results_dir: "./results"
  dcd_metrics_file: "dcd_metrics.txt"
  naive_rag_metrics_file: "naive_rag_metrics.txt"

openai:
  base_url: "your_api_url"
  api_key: "your_api_key"
  model_name: "your_model"
  temperature: 0.0

context_eval:
  temperature: 0.1
```

No hardcoded values — all settings are in the YAML file.
