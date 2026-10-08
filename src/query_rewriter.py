"""
When retrieved evidence is graded weak/irrelevant, blindly retrying the same
query against the same index returns the same weak results. Instead, this
node asks the LLM to rewrite the question into a version better suited to
vector similarity search (more explicit, disambiguated, keyword-rich).
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.llm import get_grader_llm

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite user questions into better search queries for a "
            "vector similarity retriever over a document corpus. Look at the "
            "input question and infer its underlying semantic intent. "
            "Produce ONE rewritten question that is more specific, "
            "disambiguated, and keyword-rich than the original -- but do not "
            "change what is actually being asked. Return only the rewritten "
            "question, nothing else.",
        ),
        ("human", "Original question: {question}"),
    ]
)


def rewrite_query(question: str) -> str:
    llm = get_grader_llm()
    chain = _REWRITE_PROMPT | llm | StrOutputParser()
    rewritten = chain.invoke({"question": question}).strip()
    return rewritten or question
