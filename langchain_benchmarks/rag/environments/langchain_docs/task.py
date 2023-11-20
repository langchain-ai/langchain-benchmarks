from langchain_benchmarks.rag.environments.langchain_docs import (
    langchain_docs_retriever,
    architectures,
)
from langchain_benchmarks.schema import RetrievalTask

DATASET_ID = (
    "452ccafc-18e1-4314-885b-edd735f17b9d"  # ID of public LangChain Docs dataset
)

LANGCHAIN_DOCS_TASK = RetrievalTask(
    name="LangChain Docs Q&A",
    dataset_id=DATASET_ID,
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
