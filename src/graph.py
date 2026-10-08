"""
Wires every node from grading.py / generation.py / verification.py /
query_rewriter.py / web_search.py into a LangGraph StateGraph with the
conditional edges that make this "self-corrective" rather than a straight
line pipeline.

Graph shape (see README.md for the ASCII diagram):

    retrieve -> grade_documents -> {generate | transform_query | web_search}
    generate -> grade_hallucination -> {grade_answer | generate (retry)}
    grade_answer -> {END | transform_query (retry)}

Every corrective edge is bounded by state["retries"] vs settings.max_retries,
so the graph always terminates.
"""
from langgraph.graph import StateGraph, END

from src.state import GraphState
from src.config import settings
from src.vectorstore import get_retriever
from src.grading import grade_documents as grade_documents_fn
from src.generation import generate_answer
from src.verification import grade_hallucination, grade_answer as grade_answer_fn
from src.query_rewriter import rewrite_query
from src.web_search import web_search as web_search_fn


def _log(state: GraphState, message: str) -> list:
    trace = list(state.get("trace", []))
    trace.append(message)
    return trace


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> dict:
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "trace": _log(state, f"retrieve: got {len(documents)} chunk(s) for '{question}'"),
    }


def grade_documents_node(state: GraphState) -> dict:
    question = state["question"]
    documents = state["documents"]
    relevant = grade_documents_fn(question, documents)
    return {
        "documents": relevant,
        "trace": _log(
            state,
            f"grade_documents: {len(relevant)}/{len(documents)} chunk(s) kept as relevant",
        ),
    }


def transform_query_node(state: GraphState) -> dict:
    rewritten = rewrite_query(state["question"])
    return {
        "question": rewritten,
        "retries": state.get("retries", 0) + 1,
        "trace": _log(state, f"transform_query: rewrote question to '{rewritten}'"),
    }


def web_search_node(state: GraphState) -> dict:
    docs = web_search_fn(state["question"])
    combined = state.get("documents", []) + docs
    return {
        "documents": combined,
        "used_web_search": True,
        "trace": _log(state, f"web_search: added {len(docs)} web result(s)"),
    }


def generate_node(state: GraphState) -> dict:
    answer = generate_answer(state["question"], state["documents"])
    return {
        "generation": answer,
        "trace": _log(state, "generate: produced draft answer"),
    }


def grade_hallucination_node(state: GraphState) -> dict:
    ok = grade_hallucination(state["documents"], state["generation"])
    update = {
        "hallucination_ok": ok,
        "trace": _log(
            state, f"grade_hallucination: {'grounded' if ok else 'NOT grounded'}"
        ),
    }
    if not ok:
        # A failed check sends the graph back to `generate`. Count that as a
        # retry, otherwise regenerate -> check -> regenerate could repeat
        # without the retry counter ever moving, and the loop would not be
        # bounded by MAX_RETRIES like every other corrective loop.
        update["retries"] = state.get("retries", 0) + 1
    return update


def grade_answer_node(state: GraphState) -> dict:
    ok = grade_answer_fn(state["original_question"], state["generation"])
    return {
        "answers_question": ok,
        "trace": _log(
            state, f"grade_answer: {'answers question' if ok else 'off-target'}"
        ),
    }


def give_up_node(state: GraphState) -> dict:
    """Reached only if we exhaust retries without a fully-verified answer.
    Returns an honest, hedged response instead of silently shipping a
    possibly-ungrounded or non-responsive answer."""
    hedge = (
        "I wasn't able to find and verify strong enough evidence in the "
        "available documents to answer this confidently. Here's my best "
        f"attempt based on what was retrieved:\n\n{state.get('generation', '(no answer generated)')}\n\n"
        "Treat the above as provisional -- it did not pass the automated "
        "evidence-verification check."
    )
    return {
        "generation": hedge,
        "trace": _log(state, "give_up: exhausted retries, returning hedged answer"),
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def decide_after_grading(state: GraphState) -> str:
    """After grade_documents: do we have enough relevant evidence to
    generate, or do we need to correct the retrieval first?"""
    has_evidence = len(state["documents"]) > 0
    retries = state.get("retries", 0)

    if has_evidence:
        return "generate"
    if retries >= settings.max_retries:
        # Out of retries -- try web search once as a last resort if available,
        # otherwise generate anyway (generation.py will say evidence is thin).
        return "web_search" if settings.web_search_enabled and not state.get("used_web_search") else "generate"
    return "transform_query"


def decide_after_hallucination_check(state: GraphState) -> str:
    if state["hallucination_ok"]:
        return "grade_answer"
    if state.get("retries", 0) >= settings.max_retries:
        return "give_up"
    return "generate"  # regenerate from the same evidence, LLM temperature will vary output


def decide_after_answer_check(state: GraphState) -> str:
    if state["answers_question"]:
        return "end"
    if state.get("retries", 0) >= settings.max_retries:
        return "give_up"
    return "transform_query"  # the evidence was grounded but didn't answer the question -- try different evidence


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_hallucination", grade_hallucination_node)
    workflow.add_node("grade_answer", grade_answer_node)
    workflow.add_node("give_up", give_up_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )

    # after rewriting the query, go back to retrieval with the new query
    workflow.add_edge("transform_query", "retrieve")
    # after web search augments documents, go straight to generation
    workflow.add_edge("web_search", "generate")

    workflow.add_edge("generate", "grade_hallucination")

    workflow.add_conditional_edges(
        "grade_hallucination",
        decide_after_hallucination_check,
        {
            "grade_answer": "grade_answer",
            "generate": "generate",
            "give_up": "give_up",
        },
    )

    workflow.add_conditional_edges(
        "grade_answer",
        decide_after_answer_check,
        {
            "end": END,
            "transform_query": "transform_query",
            "give_up": "give_up",
        },
    )

    workflow.add_edge("give_up", END)

    return workflow.compile()


def run(question: str) -> GraphState:
    graph = build_graph()
    initial_state: GraphState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "used_web_search": False,
        "retries": 0,
        "hallucination_ok": None,
        "answers_question": None,
        "trace": [],
    }
    final_state = graph.invoke(initial_state)
    return final_state