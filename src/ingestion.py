"""
Loads documents from a folder (.txt, .md, .pdf) and splits them into chunks
suitable for embedding. Keep this module dumb on purpose: it has one job,
turn files into a List[Document], nothing about embeddings or the vector
store lives here.
"""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)

from src.config import settings

LOADER_BY_SUFFIX = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".pdf": PyPDFLoader,
}


def load_documents(source_dir: str) -> List[Document]:
    """Walk `source_dir` and load every supported file into Document objects."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    docs: List[Document] = []
    for file_path in sorted(source_path.rglob("*")):
        if not file_path.is_file():
            continue
        loader_cls = LOADER_BY_SUFFIX.get(file_path.suffix.lower())
        if loader_cls is None:
            continue  # silently skip unsupported file types
        try:
            loaded = loader_cls(str(file_path)).load()
        except Exception as exc:  # noqa: BLE001 - we want ingestion to be resilient
            print(f"[ingestion] skipped {file_path.name}: {exc}")
            continue
        for d in loaded:
            d.metadata["source"] = file_path.name
        docs.extend(loaded)

    if not docs:
        raise ValueError(
            f"No supported documents (.txt/.md/.pdf) found under {source_dir}"
        )
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split loaded documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def load_and_chunk(source_dir: str) -> List[Document]:
    docs = load_documents(source_dir)
    chunks = chunk_documents(docs)
    print(f"[ingestion] loaded {len(docs)} document(s) -> {len(chunks)} chunk(s)")
    return chunks
