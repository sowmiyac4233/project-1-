"""
Grades each retrieved document for relevance to the question. This is the
first line of defense against hallucination: if we filter out weak/irrelevant
chunks *before* generation, the generator LLM has less room to make things up
from noise.
"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.llm import get_grader_llm


class DocumentGradeItem(BaseModel):
    """Relevance grade for one document, identified by its position in the
    batch (1-indexed) so the LLM's answers can be matched back to the
    original document list."""

    document_number: int = Field(description="The 1-indexed number of the document being graded.")
    binary_score: str = Field(
        description="Is this document relevant to the question? Answer exactly 'yes' or 'no'."
    )


class DocumentGradeBatch(BaseModel):
    """A grade for every document in one batch, so we can grade a whole
    retrieved set in a single LLM call instead of one call per document --
    this cuts both token usage and latency roughly N-fold for N documents,
    which matters a lot on rate-limited free API tiers."""

    grades: List[DocumentGradeItem]


_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict grader assessing whether retrieved documents "
            "contain information relevant to a user's question. This is not "
            "about exact keyword overlap -- if a document has semantic "
            "content that would help answer the question, grade it 'yes'. "
            "If a document is off-topic or would not help, grade it 'no'. "
            "Be strict: when in doubt, prefer 'no' over letting weak evidence "
            "through, since a downstream system will generate an answer only "
            "from documents you mark 'yes'.\n\n"
            "You will be given several numbered documents. Return exactly "
            "one grade per document, using the same document_number.",
        ),
        (
            "human",
            "User question: {question}\n\nDocuments:\n\n{documents}",
        ),
    ]
)


def _format_numbered_docs(documents: List[Document]) -> str:
    return "\n\n".join(
        f"[Document {i}]\n{doc.page_content}" for i, doc in enumerate(documents, start=1)
    )


def grade_documents(question: str, documents: List[Document]) -> List[Document]:
    """Return only the documents graded relevant to `question`.

    Grades the whole batch of retrieved documents in a single LLM call
    (rather than one call per document) to cut token usage and latency.

    If the batched grading call itself fails (malformed structured output,
    rate limit, timeout, etc.), we fail safe by treating every document in
    the batch as "not relevant" -- letting ungraded documents through to
    generation is exactly the hallucination risk this step exists to
    prevent."""
    if not documents:
        return []

    llm = get_grader_llm().with_structured_output(DocumentGradeBatch)
    chain = _GRADE_PROMPT | llm

    try:
        result: DocumentGradeBatch = chain.invoke(
            {"question": question, "documents": _format_numbered_docs(documents)}
        )
    except Exception as exc:  # noqa: BLE001 -- grader output validation errors, timeouts, etc.
        print(f"[grading] batched grader call failed, defaulting all to 'not relevant': {exc}")
        return []

    grade_by_number = {g.document_number: g.binary_score for g in result.grades}

    relevant_docs: List[Document] = []
    for i, doc in enumerate(documents, start=1):
        score = grade_by_number.get(i, "no")  # missing grade -> safe default of 'no'
        if score.strip().lower().startswith("y"):
            relevant_docs.append(doc)
    return relevant_docs