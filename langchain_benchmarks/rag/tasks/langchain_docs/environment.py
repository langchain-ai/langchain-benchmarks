from functools import partial

from langchain_benchmarks.rag.tasks.langchain_docs.architectures import ARCH_FACTORIES
from langchain_benchmarks.rag.tasks.langchain_docs.indexing import (
    RETRIEVER_FACTORIES,
)
from langchain_benchmarks.rag.tasks.langchain_docs.indexing.retriever_registry import (
    DOCS_FILE,
    load_docs_from_parquet,
)
from langchain_benchmarks.schema import RetrievalEnvironment


def create_environment() -> RetrievalEnvironment:
    """Create a retrieval environment."""
    return RetrievalEnvironment(
        retriever_factories=RETRIEVER_FACTORIES,
        architecture_factories=ARCH_FACTORIES,
        get_docs=partial(load_docs_from_parquet, DOCS_FILE),
    )
