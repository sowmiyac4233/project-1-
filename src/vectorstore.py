"""
Wraps Chroma persistence + embedding-model selection so the rest of the
codebase never has to know whether we're using OpenAI or local embeddings.
"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import settings

# Cached so the (potentially slow-to-load, e.g. local HuggingFace) embedding
# model and vector store connection are only created once per process,
# instead of being reloaded from disk on every single retrieve() call in
# the graph -- each question can trigger several retrieval calls (one per
# corrective retry), so without this cache the same model gets reloaded
# repeatedly within a single run.
_embeddings_cache = None
_vectorstore_cache = None


def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    if settings.embedding_provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        _embeddings_cache = HuggingFaceEmbeddings(model_name=settings.local_embedding_model)
    else:
        from langchain_openai import OpenAIEmbeddings

        _embeddings_cache = OpenAIEmbeddings(api_key=settings.openai_api_key)
    return _embeddings_cache


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Embed `chunks` and persist them, overwriting any existing collection
    of the same name so re-running ingestion doesn't duplicate content."""
    global _vectorstore_cache
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    # Clear any previous run's data for this collection before re-adding.
    try:
        existing_ids = vectorstore.get()["ids"]
        if existing_ids:
            vectorstore.delete(ids=existing_ids)
    except Exception:
        pass  # collection didn't exist yet -- nothing to clear

    vectorstore.add_documents(chunks)
    _vectorstore_cache = vectorstore  # freshly (re)built, so update the cache too
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an already-persisted vector store (call after `ingest` has run)."""
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache

    embeddings = get_embeddings()
    _vectorstore_cache = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    return _vectorstore_cache


def get_retriever(vectorstore: Optional[Chroma] = None, k: Optional[int] = None):
    vectorstore = vectorstore or load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k or settings.retrieval_k})