from typing import Iterable

from langchain.schema.document import Document

from langchain_benchmarks.rag.tasks.langchain_docs import architectures, indexing
from langchain_benchmarks.rag.tasks.langchain_docs.indexing.retriever_registry import (
    DOCS_FILE,
    load_docs_from_parquet,
)
from langchain_benchmarks.schema import RetrievalTask

# URL of public LangChain Docs dataset
DATASET_ID = "https://smith.langchain.com/public/452ccafc-18e1-4314-885b-edd735f17b9d/d"


def load_cached_docs() -> Iterable[Document]:
    """Load the docs from the cached file."""
    return load_docs_from_parquet(DOCS_FILE)


LANGCHAIN_DOCS_TASK = RetrievalTask(
    name="LangChain Docs Q&A",
    dataset_id=DATASET_ID,
    retriever_factories=indexing.RETRIEVER_FACTORIES,
    architecture_factories=architectures.ARCH_FACTORIES,
    get_docs=load_cached_docs,
    description=(
        """\
Questions and answers based on a snapshot of the LangChain python docs.

The environment provides the documents and the retriever information.

Each example is composed of a question and reference answer.

Success is measured based on the accuracy of the answer relative to the reference answer.
We also measure the faithfulness of the model's response relative to the retrieved documents (if any).
"""  # noqa: E501
    ),
)
