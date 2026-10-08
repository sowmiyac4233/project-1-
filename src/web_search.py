"""
Fallback retrieval source for when the local vector store doesn't have
enough relevant evidence even after query rewriting + retries. Optional --
if no TAVILY_API_KEY is set, this node is skipped entirely by the graph
(see src/graph.py::decide_after_grading).
"""
from typing import List
from langchain_core.documents import Document

from src.config import settings


def web_search(question: str, max_results: int = 4) -> List[Document]:
    if not settings.web_search_enabled:
        return []

    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    results = client.search(query=question, max_results=max_results)

    docs: List[Document] = []
    for r in results.get("results", []):
        docs.append(
            Document(
                page_content=r.get("content", ""),
                metadata={"source": r.get("url", "web"), "web_search": True},
            )
        )
    return docs
