"""LLM client wrapper using OpenAI API (supports compatible providers via base_url)."""

from openai import OpenAI
from pydantic import BaseModel


class LLM:
    """Client for text and structured generation using OpenAI API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        """Generate plain text completion. Returns the assistant message content."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        msg = response.choices[0].message
        if not msg.content:
            return ""
        return msg.content

    def generate_structured(self, prompt: str, response_model: type[BaseModel]) -> BaseModel:
        """Generate completion that conforms to the Pydantic model (Structured Output)."""
        completion = self._client.chat.completions.parse(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Structured response was empty or could not be parsed")
        return parsed
