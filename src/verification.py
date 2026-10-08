"""
The two self-correction checks that give this pipeline its name:

1. grade_hallucination -- is every claim in the generated answer actually
   grounded in the retrieved documents? (catches the LLM drifting/inventing
   facts even when given good evidence)
2. grade_answer -- does the generated answer actually address the question
   that was asked? (catches technically-grounded-but-non-responsive answers)

These are deliberately separate checks because they catch different failure
modes and a system could pass one while failing the other.
"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_grader_llm


class BinaryGrade(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")
    reasoning: str = Field(description="One sentence explaining the grade.")


_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a fact-checking grader. You will be given a set of "
            "source documents and a generated answer. Determine whether the "
            "answer is fully grounded in / supported by the facts in the "
            "documents. Grade 'yes' only if EVERY factual claim in the "
            "answer is traceable to the documents. Grade 'no' if the answer "
            "contains any claim not supported by, or contradicted by, the "
            "documents.",
        ),
        (
            "human",
            "Source documents:\n{documents}\n\nGenerated answer:\n{generation}",
        ),
    ]
)

_ANSWER_QUALITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether a generated answer actually "
            "addresses the user's question. Grade 'yes' if the answer is "
            "responsive to what was asked (even if it says evidence is "
            "incomplete). Grade 'no' if the answer is off-topic, evasive "
            "without cause, or answers a different question than the one "
            "asked.",
        ),
        (
            "human",
            "Question: {question}\n\nGenerated answer:\n{generation}",
        ),
    ]
)


def _format_docs(documents: List[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in documents)


def grade_hallucination(documents: List[Document], generation: str) -> bool:
    """Returns False (fails safe) if the grader call itself errors -- treating
    an unverifiable answer as "not grounded" is the conservative choice,
    since the whole point of this check is to catch answers we can't trust."""
    llm = get_grader_llm().with_structured_output(BinaryGrade)
    chain = _HALLUCINATION_PROMPT | llm
    try:
        grade: BinaryGrade = chain.invoke(
            {"documents": _format_docs(documents), "generation": generation}
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[verification] hallucination grader call failed, defaulting to 'not grounded': {exc}")
        return False
    return grade.binary_score.strip().lower().startswith("y")


def grade_answer(question: str, generation: str) -> bool:
    """Returns False (fails safe) if the grader call itself errors -- same
    reasoning as grade_hallucination above."""
    llm = get_grader_llm().with_structured_output(BinaryGrade)
    chain = _ANSWER_QUALITY_PROMPT | llm
    try:
        grade: BinaryGrade = chain.invoke({"question": question, "generation": generation})
    except Exception as exc:  # noqa: BLE001
        print(f"[verification] answer grader call failed, defaulting to 'off-target': {exc}")
        return False
    return grade.binary_score.strip().lower().startswith("y")