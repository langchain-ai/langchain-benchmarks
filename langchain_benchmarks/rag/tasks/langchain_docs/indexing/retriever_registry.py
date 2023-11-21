import logging
import os
from typing import Callable, Iterable, Optional

from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores.chroma import Chroma

from langchain_benchmarks.rag.utils.indexing import (
    get_hyde_retriever,
    get_parent_document_retriever,
    get_vectorstore_retriever,
)

logger = logging.getLogger(__name__)
_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
# Stores the scraped documents from the langchain docs website, week of 2023-11-12
REMOTE_DOCS_FILE = "https://storage.googleapis.com/benchmarks-artifacts/langchain-docs-benchmarking/docs.parquet"
DOCS_FILE = os.path.join(_DIRECTORY, "db_docs/docs.parquet")

_DEFAULT_SEARCH_KWARGS = {"k": 6}


def load_docs_from_parquet(filename: Optional[str] = None) -> Iterable[Document]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "Please install pandas to use the langchain docs benchmarking task.\n"
            "pip install pandas"
        )
    df = pd.read_parquet(filename)
    docs_transformed = [Document(**row) for row in df.to_dict(orient="records")]
    for doc in docs_transformed:
        for k, v in doc.metadata.items():
            if v is None:
                doc.metadata[k] = ""
        if not doc.page_content.strip():
            continue
        yield doc


def _chroma_retriever_factory(
    embedding: Embeddings,
    search_kwargs: Optional[dict] = None,
    transform_docs: Optional[Callable] = None,
    transformation_name: Optional[str] = None,
) -> BaseRetriever:
    docs = load_docs_from_parquet(DOCS_FILE)
    embedding_name = embedding.__class__.__name__
    vectorstore = Chroma(
        collection_name=f"langchain-benchmarks-classic-{embedding_name}",
        embedding_function=embedding,
        persist_directory="./chromadb",
    )
    return get_vectorstore_retriever(
        docs,
        embedding,
        vectorstore,
        collection_name="langchain-docs",
        transform_docs=transform_docs,
        transformation_name=transformation_name,
        search_kwargs=search_kwargs or _DEFAULT_SEARCH_KWARGS,
    )


def _chroma_parent_document_retriever_factory(
    embedding: Embeddings,
    search_kwargs: Optional[dict] = None,
) -> BaseRetriever:
    docs = load_docs_from_parquet(DOCS_FILE)
    embedding_name = embedding.__class__.__name__
    vectorstore = Chroma(
        collection_name=f"langchain-benchmarks-parent-doc-{embedding_name}",
        embedding_function=embedding,
        persist_directory="./chromadb",
    )
    return get_parent_document_retriever(
        docs,
        embedding,
        vectorstore,
        collection_name="langchain-docs",
        search_kwargs=search_kwargs or _DEFAULT_SEARCH_KWARGS,
    )


def _chroma_hyde_retriever_factory(
    embedding: Embeddings,
    search_kwargs: Optional[dict] = None,
) -> BaseRetriever:
    docs = load_docs_from_parquet(DOCS_FILE)
    embedding_name = embedding.__class__.__name__
    vectorstore = Chroma(
        collection_name=f"langchain-benchmarks-hyde-{embedding_name}",
        embedding_function=embedding,
        persist_directory="./chromadb",
    )
    return get_hyde_retriever(
        docs,
        embedding,
        vectorstore,
        collection_name="langchain-docs",
        search_kwargs=search_kwargs or _DEFAULT_SEARCH_KWARGS,
    )


RETRIEVER_FACTORIES = {
    "basic": _chroma_retriever_factory,
    "parent-doc": _chroma_parent_document_retriever_factory,
    "hyde": _chroma_hyde_retriever_factory,
}
