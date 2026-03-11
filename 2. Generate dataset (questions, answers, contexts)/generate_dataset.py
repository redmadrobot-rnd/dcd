"""
Build RAG dataset from existing .txt documents.

- Does NOT generate documents (reads ready .txt from output/)
- Generates Q&A pairs via LLM for each document
- Writes dataset.xlsx, dataset_classification.json, metadata_mapping.json
"""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from llm import LLM
from metadata_mapping import (
    build_metadata_mapping_from_documents,
    format_question_prefix,
    get_collection_display_name,
    get_dataset_config,
    get_domain_display_name,
)
from prompts import build_qa_prompt
from schemas import QAList
from utils import (
    append_rows_to_dataset_xlsx,
    append_to_classification_json,
    classification_json_path,
    dataset_xlsx_path,
    read_document,
    write_metadata_mapping_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    """Load YAML config."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("output_dir", "output")
    return data


def iter_documents_from_output(output_dir: Path):
    """
    Yield (domain_name, collection_name, document_name, doc_path) for each .txt in output/.
    """
    if not output_dir.is_dir():
        return

    for domain_path in sorted(output_dir.iterdir()):
        if not domain_path.is_dir():
            continue
        domain_name = domain_path.name

        for collection_path in sorted(domain_path.iterdir()):
            if not collection_path.is_dir():
                continue
            collection_name = collection_path.name

            for doc_path in sorted(collection_path.glob("*.txt")):
                document_name = doc_path.stem
                yield domain_name, collection_name, document_name, doc_path


def main() -> None:
    load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set in .env")
        raise SystemExit(1)

    output_dir = "../output"
    output_path = Path(__file__).resolve().parent / output_dir

    # Clear previous dataset (fresh build from documents)
    for f in (dataset_xlsx_path(output_path), classification_json_path(output_path)):
        if f.is_file():
            f.unlink()
            logger.info("Cleared existing %s", f.name)

    # Build config-like structure for metadata from domains_variables + templates_config
    doc_config = build_metadata_mapping_from_documents(output_path)
    write_metadata_mapping_json(output_path, doc_config["metadata_mapping"])
    logger.info("Wrote metadata_mapping.json for RAG filter keys")

    # Use the built config for display names
    doc_config_for_display = doc_config.get("config", {})

    dataset_config = get_dataset_config()
    qa_pairs_per_document = dataset_config.get("qa_pairs_per_document", 2)

    llm = LLM(base_url=base_url, api_key=api_key, model=model)

    processed = 0
    for domain_name, collection_name, document_name, doc_path in iter_documents_from_output(
        output_path
    ):
        doc_text = read_document(doc_path)
        if not doc_text.strip():
            logger.warning("Document is empty, skipping: %s", doc_path)
            continue

        logger.info("Processing: %s / %s / %s", domain_name, collection_name, document_name)

        try:
            prompt_qa = build_qa_prompt(doc_text, qa_pairs_per_document)
            qa_list = llm.generate_structured(prompt_qa, QAList)
        except Exception as e:
            logger.exception("Failed to generate Q&A for %s: %s", document_name, e)
            continue

        domain_display = get_domain_display_name(doc_config_for_display, domain_name)
        collection_display = get_collection_display_name(
            doc_config_for_display, domain_name, collection_name
        )
        prefix = format_question_prefix(domain_display, collection_display)

        rows = []
        classification_entries = []
        for item in qa_list.items:
            question_with_prefix = prefix + item.question
            rows.append(
                (
                    domain_name,
                    collection_name,
                    document_name,
                    question_with_prefix,
                    item.answer,
                    item.context,
                )
            )
            classification_entries.append(
                {
                    "question": question_with_prefix,
                    "answer": item.answer,
                    "context": item.context,
                    "domain": domain_name,
                    "collection": collection_name,
                    "document": document_name,
                }
            )

        append_rows_to_dataset_xlsx(output_path, rows)
        append_to_classification_json(output_path, classification_entries)
        processed += 1
        logger.info("Appended %d Q&A pairs for %s", len(rows), document_name)

    logger.info("Done. Processed %d documents. Output dir: %s", processed, output_path.resolve())
    logger.info("Files: dataset.xlsx, dataset_classification.json, metadata_mapping.json")


if __name__ == "__main__":
    main()
