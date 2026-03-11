import os
from typing import List, Literal

from pydantic import BaseModel, Field

from config import client_openai


def get_domains_model(domains: List[str]):
    class Domain(BaseModel):
        """Model for determining the type of residential complex (RC)"""
        reasoning: str = Field(..., description="Reasoning for choosing a RC")
        domain: Literal[*domains] = Field(..., description="RC on which the question is asked")
    return Domain

def get_collections_model(collections: List[str]):
    class Collection(BaseModel):
        """Model for determining the section"""
        reasoning: str = Field(..., description="Reasoning for choosing the section")
        collection: Literal[*collections] = Field(..., description="Section on which the question is asked")
        
    return Collection
    
def create_classification_schema(domains, collections):
    """Build Pydantic models for classifying a query into domain and collection (with Literal choices)."""
    domains_scheme = get_domains_model(domains)
    collection_scheme = get_collections_model(collections)
        
    return domains_scheme, collection_scheme


def classification_by_so(text: str, scheme: BaseModel, system_prompt: str):
    response = client_openai.beta.chat.completions.parse(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {text}"}
        ],
        response_format=scheme,
    )
    return response.choices[0].message.parsed


def generate_answer(query_text: str, context: str):
    response = client_openai.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role": "system", "content": "Answer the question using only the context provided."},
            {"role": "user", "content": f"Question: {query_text}. Context: {context}. Answer:"}
        ]
    )
    return response.choices[0].message.content