import logging
import os
from functools import partial
from typing import Callable, Iterable, List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.indexes import SQLRecordManager, index
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.schema.storage import BaseStore
from langchain.schema.vectorstore import VectorStore
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
COLLECTION_NAME = "langchain-docs"
RECORD_MANAGER_DB_URL = (
    os.environ.get("RECORD_MANAGER_DB_URL")
    or f"sqlite:///{_DIRECTORY}_record_manager.sql"
)


def transform_docs_parent_child(
    documents: Iterable[Document],
    child_splitter: TextSplitter,
    docstore: BaseStore,
    id_key: str,
    *,
    parent_splitter: Optional[TextSplitter] = None,
) -> Iterable[Document]:
    """Transforms documents into child <-> parent documents."""
    if parent_splitter is not None:
        documents = parent_splitter.split_documents(documents)
    doc_ids = []
    for doc in documents:
        yield doc
        _id = doc.metadata[id_key]
        doc_ids.append((_id, doc))
        sub_docs = child_splitter.split_documents([doc])
        for _doc in sub_docs:
            _doc.metadata[id_key] = _id
            yield _doc
    docstore.mset(doc_ids)


def _default_hyde_embedder():
    class hypotheticalQuestions(BaseModel):
        """Write user queries that could be answered by the document."""

        questions: List[str]

    return (
        (
            {"doc": lambda x: x.page_content}
            # Only asking for 3 hypothetical questions, but this could be adjusted
            | ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an AI creating an inverted index for"
                        " document retrieval. "
                        "Analyze the content of the following document and generate"
                        " relevant user queries "
                        "that would likely retrieve this document. Document content:"
                        "\n\n```document.txt\n{doc}\n```",
                    ),
                    (
                        "user",
                        "Based on the document's content, what specific technical queries"
                        " or questions"
                        " are users likely to search that this document can answer?",
                    ),
                ]
            )
            | ChatOpenAI(max_retries=0, model="gpt-4-1106-preview").bind_functions(
                functions=[hypotheticalQuestions],
                function_call="hypotheticalQuestions",
            )
            | JsonKeyOutputFunctionsParser(key_name="questions")
        )
        .with_retry(stop_after_attempt=3)
        .with_config(
            run_name="HyDE",
            metadata={"benchmark_environment": "langchain-docs"},
            tags=["langchain-benchmarks"],
        )
    )


def transform_docs_hyde(
    documents: Iterable[Document],
    docstore: BaseStore,
    id_key: str,
    *,
    query_generator: Optional[Runnable] = None,
    runnable_config: Optional[RunnableConfig] = None,
) -> Iterable[Document]:
    """Generates hypothetical document embeddings."""
    if query_generator is None:
        query_generator = _default_hyde_embedder()
        logger.info(f"Using default query generator\n{query_generator}")
    generator = query_generator or _default_hyde_embedder()
    docs = list(documents)
    questions = generator.batch(
        docs, runnable_config or {"max_concurrency": 5}, return_exceptions=True
    )
    doc_ids = []
    for doc, expansions in zip(documents, questions):
        yield doc
        if isinstance(expansions, BaseException):
            logger.error(
                f"Error generating questions for document {doc.metadata[id_key]}"
            )
            continue
        expansion_docs = [
            Document(page_content=s, metadata=doc.metadata) for s in expansions
        ]
        yield from expansion_docs
        doc_ids.append((doc.metadata[id_key], doc))
    docstore.mset(doc_ids)


def create_index(
    docs: Iterable[Document],
    embedding: Embeddings,
    vectorstore: VectorStore,
    collection_name: str,
    *,
    transform_docs: Optional[Callable] = None,
    transformation_name: Optional[str] = None,
):
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
        f"{vectorstore_name}/{collection_name}_{vectorstore_name}_{embedding_name}_{transformation_name}",
        db_url=RECORD_MANAGER_DB_URL,
    )
    record_manager.create_schema()
    return index(
        tqdm(transformed_docs),
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
    )


def get_vectorstore_retriever(
    docs: Iterable[Document],
    embedding: Embeddings,
    vectorstore: VectorStore,
    collection_name: str,
    *,
    transform_docs: Optional[Callable] = None,
    transformation_name: Optional[str] = None,
    search_kwargs: Optional[dict] = None,
) -> BaseRetriever:
    """Index the documents (with caching) and return a vector store retriever."""
    index_stats = create_index(
        docs,
        embedding,
        vectorstore,
        collection_name=collection_name,
        transform_docs=transform_docs,
        transformation_name=transformation_name,
    )
    logger.info(f"Index stats: {index_stats}")
    kwargs = search_kwargs or {}
    kwargs.setdefault("metadata", {}).setdefault(
        "benchmark_environment", "langchain-docs"
    )
    kwargs.setdefault("tags", []).append("langchain-benchmarks")
    return vectorstore.as_retriever(**kwargs)


def get_parent_document_retriever(
    docs: Iterable[Document],
    embedding: Embeddings,
    vectorstore: VectorStore,
    collection_name: str,
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
        docs,
        embedding,
        vectorstore,
        collection_name=collection_name,
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
        search_kwargs=search_kwargs or {},
        id_key=id_key,
    )


def get_hyde_retriever(
    docs: Iterable[Document],
    embedding: Embeddings,
    vectorstore: VectorStore,
    collection_name: str,
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
        docs,
        embedding,
        vectorstore,
        collection_name=collection_name,
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
        search_kwargs=search_kwargs,
        metadata=metadata,
        tags=["langchain-benchmarks"],
    )
