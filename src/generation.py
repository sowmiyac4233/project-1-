"""
Generates an answer strictly from the (already graded-relevant) retrieved
documents. The prompt explicitly forbids using outside knowledge, since the
whole point of the pipeline is that the answer must be traceable to evidence
that passed the relevance grader.
"""
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.llm import get_generator_llm

_GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful assistant answering questions using ONLY the "
            "provided context. Rules:\n"
            "1. Every factual claim you make must be directly supported by "
            "the context below.\n"
            "2. If the context is insufficient to fully answer the question, "
            "say so explicitly rather than filling gaps from general "
            "knowledge.\n"
            "3. Do not mention 'the context' or 'the documents' in your "
            "answer -- write as if you simply know this, but never claim "
            "something the context doesn't support.\n"
            "4. Be concise and directly answer the question asked.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)


def _format_docs(documents: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in documents
    )


def generate_answer(question: str, documents: List[Document]) -> str:
    llm = get_generator_llm(temperature=0.2)
    chain = _GENERATE_PROMPT | llm | StrOutputParser()
    return chain.invoke(
        {"context": _format_docs(documents), "question": question}
    ).strip()
