"""
Unit tests focused on the *logic* of the pipeline -- routing decisions and
node behavior -- using mocked LLM/grader calls so these run fast, free, and
without an API key. They do NOT test real retrieval quality or real LLM
grading accuracy; that's an integration/eval concern, not a unit test.

Run with:
    pytest tests/test_pipeline.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch
from langchain_core.documents import Document

from src.state import GraphState
from src.graph import (
    decide_after_grading,
    decide_after_hallucination_check,
    decide_after_answer_check,
    grade_hallucination_node,
)
from src.config import settings


def _base_state(**overrides) -> GraphState:
    state: GraphState = {
        "question": "What triggers query rewriting?",
        "original_question": "What triggers query rewriting?",
        "documents": [],
        "used_web_search": False,
        "retries": 0,
        "hallucination_ok": None,
        "answers_question": None,
        "trace": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Routing: after document grading
# ---------------------------------------------------------------------------

def test_routes_to_generate_when_relevant_docs_exist():
    state = _base_state(documents=[Document(page_content="relevant chunk")])
    assert decide_after_grading(state) == "generate"


def test_routes_to_transform_query_when_no_docs_and_retries_remain():
    state = _base_state(documents=[], retries=0)
    assert decide_after_grading(state) == "transform_query"


def test_routes_to_web_search_when_retries_exhausted_and_web_enabled():
    state = _base_state(documents=[], retries=settings.max_retries)
    with patch.object(type(settings), "web_search_enabled", property(lambda self: True)):
        assert decide_after_grading(state) == "web_search"


def test_routes_to_generate_when_retries_exhausted_and_web_disabled():
    state = _base_state(documents=[], retries=settings.max_retries)
    with patch.object(type(settings), "web_search_enabled", property(lambda self: False)):
        assert decide_after_grading(state) == "generate"


def test_does_not_reuse_web_search_twice():
    state = _base_state(documents=[], retries=settings.max_retries, used_web_search=True)
    with patch.object(type(settings), "web_search_enabled", property(lambda self: True)):
        # already used web search once -- should fall through to generate, not loop forever
        assert decide_after_grading(state) == "generate"


# ---------------------------------------------------------------------------
# Routing: after hallucination check
# ---------------------------------------------------------------------------

def test_routes_to_grade_answer_when_grounded():
    state = _base_state(hallucination_ok=True)
    assert decide_after_hallucination_check(state) == "grade_answer"


def test_routes_to_regenerate_when_not_grounded_and_retries_remain():
    state = _base_state(hallucination_ok=False, retries=0)
    assert decide_after_hallucination_check(state) == "generate"


def test_routes_to_give_up_when_not_grounded_and_retries_exhausted():
    state = _base_state(hallucination_ok=False, retries=settings.max_retries)
    assert decide_after_hallucination_check(state) == "give_up"


# ---------------------------------------------------------------------------
# Routing: after answer-quality check
# ---------------------------------------------------------------------------

def test_routes_to_end_when_answer_is_responsive():
    state = _base_state(answers_question=True)
    assert decide_after_answer_check(state) == "end"


def test_routes_to_transform_query_when_off_target_and_retries_remain():
    state = _base_state(answers_question=False, retries=0)
    assert decide_after_answer_check(state) == "transform_query"


def test_routes_to_give_up_when_off_target_and_retries_exhausted():
    state = _base_state(answers_question=False, retries=settings.max_retries)
    assert decide_after_answer_check(state) == "give_up"


# ---------------------------------------------------------------------------
# Bounded-termination check: if the hallucination check fails every single
# time, the generate -> check -> generate loop must still end in give_up
# after at most MAX_RETRIES failures. Uses the REAL grade_hallucination_node
# (with the LLM call patched to always say "not grounded") so the test
# fails if the node ever stops counting retries.
# ---------------------------------------------------------------------------

def test_hallucination_loop_always_terminates_within_bounded_retries():
    state = _base_state(
        documents=[Document(page_content="some evidence")],
        generation="a draft answer",
    )
    max_hops = 50  # generous ceiling; hitting it means the loop never ends
    hops = 0
    decision = "generate"

    with patch("src.graph.grade_hallucination", return_value=False):
        while decision not in ("end", "give_up") and hops < max_hops:
            hops += 1
            state.update(grade_hallucination_node(state))
            decision = decide_after_hallucination_check(state)

    assert decision == "give_up", "hallucination loop did not terminate"
    assert hops == settings.max_retries