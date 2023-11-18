from typing import Iterable, List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.document import Document
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.schema.storage import BaseStore
from langchain.text_splitter import TextSplitter
import logging

logger = logging.getLogger(__name__)


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
                        "You are an AI creating an inverted index for document retrieval. "
                        "Analyze the content of the following document and generate relevant user queries "
                        "that would likely retrieve this document. Document content:\n\n```document.txt\n{doc}\n```",
                    ),
                    (
                        "user",
                        "Based on the document's content, what specific technical queries or questions"
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
