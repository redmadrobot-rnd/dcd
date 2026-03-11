from pydantic import BaseModel, Field
from typing import Literal


class ARCVerdict(BaseModel):
    """SB ARC — relevance and completeness of the answer."""
    D:  bool = Field(description="Direct Answer: answer directly addresses the question")
    P:  bool = Field(description="Completeness: all aspects of the question are covered")
    Sp: bool = Field(description="Specificity: answer is concrete, not generic")
    V:  bool = Field(description="No Vagueness: answer is free from vague or ambiguous language")
    reasoning: str = Field(description="Brief reasoning for the verdict")


class CRVerdict(BaseModel):
    """SB CR — full use of context."""
    verdict:       bool       = Field(description="True if ALL relevant context facts are used in the answer")
    missing_facts: list[str]  = Field(description="List of relevant facts from context missing in the answer")
    reasoning:     str        = Field(description="Brief reasoning for the verdict")


class FAVerdict(BaseModel):
    """SB FA — actual accuracy of the answer."""
    supported:         bool = Field(description="All statements in the answer are supported by the context")
    no_contradictions: bool = Field(description="The answer does not contradict the context")
    no_hallucinations: bool = Field(description="The answer contains no information absent from the context")
    verdict:           bool = Field(description="True only if all three conditions above are true")
    reasoning:         str  = Field(description="Brief reasoning for the verdict")


class ContextRelevanceScore(BaseModel):
    """Evaluation of the correspondence of the found context to the original one."""
    explanation: str = Field(
        description="A brief explanation of why the rating was given. Specifically note any discrepancies in the numbers/figures, if any."
    )
    score: Literal[0, 1, 2] = Field(
        description="Score: 0 — context not found / inconsistent, 1 — context partially found (there are discrepancies), 2 — context fully found (all facts and figures are present)."
    )
