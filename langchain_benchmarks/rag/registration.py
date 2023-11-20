"""Registry of RAG environments for ease of access."""
import dataclasses
from typing import Callable, Dict, List

from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from langchain_benchmarks.rag.environments import langchain_docs
from langchain_benchmarks.rag.environments.langchain_docs import (
    architectures,
    langchain_docs_retriever,
)
from langchain_benchmarks.utils._registration import Environment, Registry


@dataclasses.dataclass(frozen=True)
class RetrievalEnvironment(Environment):
    retriever_factories: Dict[str, Callable[[Embeddings], BaseRetriever]]  # noqa: F821
    """Factories that index the docs using the specified strategy."""
    architecture_factories: Dict[
        str, Callable[[Embeddings], BaseRetriever]
    ]  # noqa: F821
    """Factories methods that help build some off-the-shelf architectures。"""

    @property
    def _table(self) -> List[List[str]]:
        table = super()._table
        return table + [
            ["Retriever Factories", ", ".join(self.retriever_factories.keys())],
            ["Architecture Factories", ", ".join(self.architecture_factories.keys())],
        ]

# Using lower case naming to make a bit prettier API when used in a notebook
registry = Registry(
    environments=[
        RetrievalEnvironment(
            id=0,
            name="LangChain Docs Q&A",
            dataset_id=langchain_docs.DATASET_ID,
            retriever_factories=langchain_docs_retriever.RETRIEVER_FACTORIES,
            architecture_factories=architectures.ARCH_FACTORIES,
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
    ]
)
