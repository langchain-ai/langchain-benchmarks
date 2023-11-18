import logging
import os
from functools import partial
from typing import Callable, Iterable, Optional

import pandas as pd
from langchain.indexes import SQLRecordManager, index
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable
from langchain.schema.storage import BaseStore
from langchain.schema.vectorstore import VectorStore
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_benchmarks.rag.utils.indexing import (
    transform_docs_hyde,
    transform_docs_parent_child,
)

from .download_db import DOCS_FILE, fetch_remote_parquet_file

logger = logging.getLogger(__name__)
_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
COLLECTION_NAME = "langchain-docs"
RECORD_MANAGER_DB_URL = (
    os.environ.get("RECORD_MANAGER_DB_URL")
    or f"sqlite:///{_DIRECTORY}_record_manager.sql"
)

_DEFAULT_SEARCH_KWARGS = {"k": 6}


def load_docs_from_parquet(filename: Optional[str] = None) -> Iterable[Document]:
    df = pd.read_parquet(filename)
    docs_transformed = [Document(**row) for row in df.to_dict(orient="records")]
    for doc in docs_transformed:
        for k, v in doc.metadata.items():
            if v is None:
                doc.metadata[k] = ""
        if not doc.page_content.strip():
            continue
        yield doc


def create_index(
    embedding: Embeddings,
    vectorstore: VectorStore,
    *,
    transform_docs: Optional[Callable] = None,
    transformation_name: Optional[str] = None,
):
    fetch_remote_parquet_file()
    docs = load_docs_from_parquet(DOCS_FILE)
    if transform_docs:
        if not transformation_name:
            raise ValueError(
                "If you provide a transform function, you must also provide a "
                "transformation name to use for the record manager."
            )
        transformed_docs = transform_docs(docs)
    else:
        transformed_docs = docs
    transformation_name = transformation_name or "raw"
    vectorstore_name = vectorstore.__class__.__name__
    embedding_name = embedding.__class__.__name__
    record_manager = SQLRecordManager(
        f"{vectorstore_name}/{COLLECTION_NAME}_{vectorstore_name}_{embedding_name}_{transformation_name}",
        db_url=RECORD_MANAGER_DB_URL,
    )
    record_manager.create_schema()

    return index(
        transformed_docs,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
    )


def get_vectorstore_retriever(
    embedding: Embeddings,
    vectorstore: VectorStore,
    *,
    transform_docs: Optional[Callable] = None,
    transformation_name: Optional[str] = None,
    search_kwargs: Optional[dict] = None,
) -> BaseRetriever:
    """Index the documents (with caching) and return a vector store retriever."""
    index_stats = create_index(
        embedding,
        vectorstore,
        transform_docs=transform_docs,
        transformation_name=transformation_name,
    )
    logger.info(f"Index stats: {index_stats}")
    kwargs = search_kwargs or _DEFAULT_SEARCH_KWARGS
    kwargs.setdefault("metadata", {}).setdefault(
        "benchmark_environment", "langchain-docs"
    )
    kwargs.setdefault("tags", []).append("langchain-benchmarks")
    return vectorstore.as_retriever(**kwargs)


def get_parent_document_retriever(
    embedding: Embeddings,
    vectorstore: VectorStore,
    *,
    child_splitter: Optional[TextSplitter] = None,
    transformation_name: Optional[str] = None,
    id_key: str = "source",
    docstore: Optional[BaseStore] = None,
    parent_splitter: Optional[TextSplitter] = None,
    search_kwargs: Optional[dict] = None,
):
    """Index the documents (with caching) and return a parent document retriever."""
    docstore = docstore or InMemoryStore()
    if child_splitter is None:
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200
        )
        transformation_name = "parent-document-recursive-cs4k_ol200"
        logger.info(f"Using default child splitter:\n{child_splitter}")
    else:
        if transformation_name is None:
            raise ValueError(
                "If you provide a custom child splitter, you must also provide a "
                "transformation name to use for the record manager."
            )
    transformation = partial(
        transform_docs_parent_child,
        child_splitter=child_splitter,
        docstore=docstore,
        parent_splitter=parent_splitter,
        id_key=id_key,
    )
    index_stats = create_index(
        embedding,
        vectorstore,
        transform_docs=transformation,
        transformation_name=transformation_name,
    )
    logger.info(f"Index stats: {index_stats}")

    return ParentDocumentRetriever(
        tags=["langchain-benchmarks"],
        metadata={"benchmark_environment": "langchain-docs"},
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs=search_kwargs or _DEFAULT_SEARCH_KWARGS,
        id_key=id_key,
    )


def get_hyde_retriever(
    embedding: Embeddings,
    vectorstore: VectorStore,
    *,
    docstore: Optional[BaseStore] = None,
    query_generator: Optional[Runnable] = None,
    id_key: str = "source",
    search_kwargs: Optional[dict] = None,
    transformation_name: Optional[str] = None,
):
    """Index the documents (with caching) and return a parent document retriever."""
    docstore = docstore or InMemoryStore()
    if query_generator is not None and transformation_name is None:
        raise ValueError(
            "If you provide a custom query generator, you must also provide a "
            "transformation name to use for the record manager."
        )
    transformation_name = transformation_name or "HyDE"
    transformation = partial(
        transform_docs_hyde,
        docstore=docstore,
        query_generator=query_generator,
        id_key=id_key,
    )
    index_stats = create_index(
        embedding,
        vectorstore,
        transform_docs=transformation,
        transformation_name=transformation_name,
    )
    logger.info(f"Index stats: {index_stats}")
    metadata = {
        "benchmark_environment": "langchain-docs",
        "retriever_stragegy": "HyDE",
        "embedding": embedding.__class__.__name__,
        "vectorstore": vectorstore.__class__.__name__,
    }

    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=id_key,
        search_kwargs=search_kwargs or _DEFAULT_SEARCH_KWARGS,
        metadata=metadata,
        tags=["langchain-benchmarks"],
    )
