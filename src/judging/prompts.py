from __future__ import annotations

CONTEXT_RELEVANCE_PROMPT = """Given the question and the retrieved context below, \
rate how relevant the context is to answering the question.
Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).
Return ONLY a decimal number, nothing else.

Question: {question}
Context: {context}
Score:"""

GROUNDEDNESS_PROMPT = """Given the context and the answer below, \
rate how well the answer is grounded in (supported by) the context.
Score from 0.0 (answer contradicts or ignores the context) to 1.0 (answer is fully supported).
Return ONLY a decimal number, nothing else.

Context: {context}
Answer: {answer}
Score:"""

ANSWER_RELEVANCE_PROMPT = """Given the question and the answer below, \
rate how relevant the answer is to the question.
Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).
Return ONLY a decimal number, nothing else.

Question: {question}
Answer: {answer}
Score:"""

PROMPTS: dict[str, str] = {
    "context_relevance": CONTEXT_RELEVANCE_PROMPT,
    "groundedness": GROUNDEDNESS_PROMPT,
    "answer_relevance": ANSWER_RELEVANCE_PROMPT,
}
