from openai import OpenAI
from models import ARCVerdict, CRVerdict, FAVerdict, ContextRelevanceScore
from config import Config
from prompts import system_prompt


class RAGMetricsEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout
        )
    
    def evaluate_arc(self, question: str, answer: str) -> ARCVerdict:
        prompt = f"""Evaluate the answer to the question using four strict binary criteria.

Question: {question}

Answer: {answer}

Criteria:
- D  (Direct Answer)  : Does the answer directly address what the question asks?
- P  (Completeness)   : Does the answer cover ALL aspects of the question without omitting key parts?
- Sp (Specificity)    : Is the answer specific and concrete (not generic)?
- V  (No Vagueness)   : Is the answer free from vague or ambiguous language?"""

        result = self.client.beta.chat.completions.parse(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a strict evaluation judge."},
                {"role": "user",   "content": prompt},
            ],
            response_format=ARCVerdict,
            temperature=self.config.temperature,
        )
        return result.choices[0].message.parsed

    def evaluate_cr(self, question: str, context: str, answer: str) -> CRVerdict:
        prompt = f"""Evaluate whether the answer uses ALL relevant facts from the context.

Context: {context}

Answer: {answer}

Steps:
1. Identify every fact in the context that is relevant to answering the question: {question}. Then check whether all such facts are used in the answer.
2. Check whether each such fact is reflected in the answer.
3. verdict = true only if ALL relevant facts are used; false if any is missing."""

        result = self.client.beta.chat.completions.parse(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a strict evaluation judge."},
                {"role": "user",   "content": prompt},
            ],
            response_format=CRVerdict,
            temperature=self.config.temperature,
        )
        return result.choices[0].message.parsed

    def evaluate_fa(self, context: str, answer: str) -> FAVerdict:
        prompt = f"""Evaluate the factual accuracy of the answer with respect to the context.

Context: {context}

Answer: {answer}

Check three conditions:
1. supported          : every statement in the answer is explicitly present in or directly entailed by the context.
2. no_contradictions  : the answer does not contradict the context.
3. no_hallucinations  : the answer contains no information absent from the context.

verdict = true only if ALL three conditions hold."""

        result = self.client.beta.chat.completions.parse(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are a strict evaluation judge."},
                {"role": "user",   "content": prompt},
            ],
            response_format=FAVerdict,
            temperature=self.config.temperature,
        )
        return result.choices[0].message.parsed
    
    def evaluate_context_relevance(self, question: str, context: str, find_context: str) -> ContextRelevanceScore:
        user_prompt = f"""Question: {question}
Original context: {context}
Found context: {find_context}"""
        
        result = self.client.beta.chat.completions.parse(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ContextRelevanceScore,
            temperature=self.config.context_temperature
        )
        return result.choices[0].message.parsed
