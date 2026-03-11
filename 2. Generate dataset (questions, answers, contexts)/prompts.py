"""Prompt templates for document generation and Q&A extraction. All text in English."""


def build_document_prompt(
    domain: str,
    collection: str,
    document_name: str,
    description: str,
) -> str:
    """Build the prompt for generating a single document (4–9 A4 pages, coherent, with numerical facts)."""
    return f"""You are an expert writer. Generate a single, coherent document in English.

Domain: {domain}
Collection: {collection}
Document title: {document_name}
Description: {description}

Requirements:
- Write in clear, connected English. No filler or vague content.
- Cover the topic thoroughly: different aspects, definitions, and practical details.
- Length: equivalent to 4 to 9 A4 pages (roughly 2000–5000 words).
- Include exact numerical facts where appropriate (dates, counts, percentages, dimensions, statistics). The document must contain at least a few concrete numbers that a reader could cite.
- Do not use placeholders like [insert X]. Output the full, final text only.
- Output only the document body. No title, no "Document:", no meta-commentary."""


def build_qa_prompt(document_text: str, number_pair: int) -> str:
    """Build the prompt for extracting 5 Q&A pairs with exact context from the document."""
    return f"""You are building a RAG evaluation dataset. Below is a document. Your task is to produce exactly {number_pair} question-answer pairs.

Rules (strict):

Base EVERYTHING only on the actual text of the document. Do not use any external knowledge or invent facts.

CRITICAL: Questions must be fact-based and specific.

Forbidden: Abstract, thematic, or overview questions (e.g., "What is the main idea of the document?", "Summarize the chapter on X", "What does the author think about Y?").

Allowed: Questions that refer to a specific detail, entity, relationship, or number present in the text. The question itself should make it clear what specific topic or entity it is asking about.

For each pair you must provide:

question: A clear, specific question that targets a concrete piece of information from the document.
answer: The answer, taken or derived strictly from the document.
context: An EXACT verbatim quote from the document that supports this answer. The quote must be a continuous passage and contain the answer.
At least half of the questions must have a numeric answer (e.g., a number, date, percentage, or quantity that appears in the document). For these, the context must contain that number.

Questions should be diverse, coming from different sections or aspects of the document.

Each question must be UNIQUE: do not ask the same or semantically equivalent question twice. Each question must target a different fact or detail from the document. If in doubt, rephrase to ask about a different sentence or number.

Output exactly {number_pair} items. No extra items, no fewer.

Document text:

{document_text}"""
