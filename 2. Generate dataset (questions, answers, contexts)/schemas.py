"""Pydantic models for Structured Output (Q&A extraction from documents)."""

from pydantic import BaseModel, field_validator

from metadata_mapping import get_dataset_config


class QAItem(BaseModel):
    """Single question-answer pair with exact context from the document."""

    question: str
    answer: str
    context: str


class QAList(BaseModel):
    """List of question-answer items per document (count from dataset_config.yaml)."""

    items: list[QAItem]

    @field_validator("items")
    @classmethod
    def items_count_from_config(cls, v: list[QAItem]) -> list[QAItem]:
        expected = get_dataset_config().get("qa_pairs_per_document", 2)
        if len(v) != expected:
            raise ValueError(f"Exactly {expected} items are required, got {len(v)}")
        return v
