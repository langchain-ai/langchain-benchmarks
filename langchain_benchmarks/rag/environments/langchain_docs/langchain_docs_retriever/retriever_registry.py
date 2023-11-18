from typing import Optional

from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.vectorstores.chroma import Chroma
from .retriever import (
    get_vectorstore_retriever,
    get_parent_document_retriever,
    get_hyde_retriever,
)


def _chroma_retriever_factory(
    embedding: Embeddings,
    search_kwargs: Optional[dict] = None,
) -> BaseRetriever:
    embedding_name = embedding.__class__.__name__
    vectorstore = Chroma(
        collection_name=f"langchain-benchmarks-classic-{embedding_name}",
        embedding_function=embedding,
        persist_directory="./chromadb",
    )
    return get_vectorstore_retriever(
        embedding,
        vectorstore,
        search_kwargs=search_kwargs,
    )


def _chroma_parent_document_retriever_factory(
    embedding: Embeddings,
    search_kwargs: Optional[dict] = None,
) -> BaseRetriever:
    embedding_name = embedding.__class__.__name__
    vectorstore = Chroma(
        collection_name=f"langchain-benchmarks-parent-doc-{embedding_name}",
        embedding_function=embedding,
        persist_directory="./chromadb",
    )
    return get_parent_document_retriever(
        embedding, vectorstore, search_kwargs=search_kwargs
    )


def _chroma_hyde_retriever_factory(
    embedding: Embeddings,
    search_kwargs: Optional[dict] = None,
) -> BaseRetriever:
    embedding_name = embedding.__class__.__name__
    vectorstore = Chroma(
        collection_name=f"langchain-benchmarks-hyde-{embedding_name}",
        embedding_function=embedding,
        persist_directory="./chromadb",
    )
    return get_hyde_retriever(embedding, vectorstore, search_kwargs=search_kwargs)


RETRIEVER_FACTORIES = {
    "basic": _chroma_retriever_factory,
    "parent-doc": _chroma_parent_document_retriever_factory,
    "hyde": _chroma_hyde_retriever_factory,
}
