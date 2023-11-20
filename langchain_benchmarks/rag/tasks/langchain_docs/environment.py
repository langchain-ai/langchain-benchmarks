from langchain_benchmarks.rag.tasks.langchain_docs.architectures import ARCH_FACTORIES
from langchain_benchmarks.rag.tasks.langchain_docs.langchain_docs_retriever import (
    RETRIEVER_FACTORIES,
)
from langchain_benchmarks.schema import RetrievalEnvironment


def create_environment() -> RetrievalEnvironment:
    """Create a retrieval environment."""
    return RetrievalEnvironment(
        retriever_factories=RETRIEVER_FACTORIES, architecture_factories=ARCH_FACTORIES
    )
