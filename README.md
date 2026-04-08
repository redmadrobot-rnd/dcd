

# DCD RAG Pipeline

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/a691bb2d-c467-4681-a878-d37a5976af75" />

End-to-end pipeline for building and evaluating a Domain-Collection-Document (DCD) RAG system with metadata filtering.

## Project Structure

### 1. Create text dataset
Generates synthetic text documents by filling templates with domain-specific variables. Produces one `.txt` file per template per domain.

**Input:** Templates, domain variables (YAML)  
**Output:** `output/<domain>/<collection>/<document>.txt`

### 2. Generate dataset (questions, answers, contexts)
Uses LLM to generate question-answer pairs from documents. Creates evaluation dataset with ground-truth contexts.

**Input:** Text documents from step 1  
**Output:** `dataset.xlsx`, `dataset_classification.json`, `metadata_mapping.json`

### 3. Create vector DB
Chunks documents, embeds with Sentence Transformers, stores in Chroma vector database with domain/collection metadata.

**Input:** Text documents from step 1  
**Output:** Chroma DB in `my_db/`, `chunks_output.json`

### 4. Inference dcd and naive rag
Runs two RAG approaches on the dataset:
- **DCD:** Classifies query → filters by metadata → reranks → generates answer
- **Naive RAG:** Queries entire DB → generates answer

**Input:** `dataset.xlsx`, Chroma DB  
**Output:** `dcd_dataset.xlsx`, `naive_rag_dataset.xlsx`

### 5. Metrics calculation
Evaluates both RAG approaches using LLM-based metrics:
- **SB ARC:** Answer Relevance & Completeness
- **SB CR:** Context Recall
- **SB FA:** Factual Accuracy
- **Context Score:** Retrieved vs. reference context quality (0-2 scale)

**Input:** Results from step 4  
**Output:** Per-sample evaluations (Excel), aggregated metrics (TXT)

## Usage

Run steps sequentially (1 → 2 → 3 → 4 → 5). Each folder contains a README with setup and execution instructions. Configure via YAML files and `.env` where needed.
