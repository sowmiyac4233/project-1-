"""
The shared state object threaded through every node in the LangGraph.
Each node reads what it needs and returns a partial dict of updates --
LangGraph merges those into this state before calling the next node.
"""
from typing import List, TypedDict, Optional
from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    # --- input ---
    question: str                  # the current question (may be rewritten mid-run)
    original_question: str         # the question as first asked, kept for the final answer-quality check

    # --- retrieval ---
    documents: List[Document]      # currently retrieved / graded-relevant documents
    used_web_search: bool          # whether the web fallback contributed to `documents`

    # --- generation ---
    generation: str                # the LLM's current draft answer

    # --- self-correction bookkeeping ---
    retries: int                   # number of corrective loops taken so far
    hallucination_ok: Optional[bool]   # None until checked; True/False after grade_hallucination
    answers_question: Optional[bool]   # None until checked; True/False after grade_answer

    # --- observability ---
    trace: List[str]               # human-readable log of which nodes fired, for --trace / debugging
